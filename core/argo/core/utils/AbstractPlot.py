from abc import abstractmethod

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import copy

# it takes a lifetime to import this (****)
# from core.argo.core.utils.argo_utils import make_list, create_list_colors

from itertools import product

import re
import os


def make_list(l):
    return l if isinstance(l, list) else [l]

# def create_list_colors(max_colors):
#     r = max_colors / 10.0
#     return plt.cm.tab10((1 / r * np.arange(10 * r)).astype(int))

class AbstractPlot():

    def __init__(self):
        pass

    def plot(self,
             panels,
             where,
             block,
             fixed_over,
             logs_directory,
             depth,
             dataset,
             tag,
             filename="log",
             dirname=".",
             figsize=(26, 10),
             legend=[],
             group_by=[],
             rcParams={},
             string_to_val={},
             labels_from_files=0,
             small_size=8,
             medium_size=10,
             bigger_size=12,
             suptitle=None,
             models_directories=[],
             degree=10,
             colors=[],
             replace_names=None):

        plt.rc('font', size=small_size)  # controls default text sizes
        plt.rc('axes', titlesize=small_size)  # fontsize of the axes title
        plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=small_size)  # fontsize of the tick labels
        plt.rc('legend', fontsize=small_size)  # legend fontsize
        plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title
        # added by ****
        matplotlib.rcParams.update(rcParams)

        self._labels_from_files = labels_from_files
        self._replace_names = replace_names
        self._colors = colors

        if suptitle is None:
            if isinstance(dataset, list):
                if len(dataset) == 1:
                    suptitle = dataset[0]
            else:
                suptitle = ' - '.join(dataset)

        # self.string_to_val = string_to_val

        if len(models_directories) > 0:
            # where, block, sources and all the rest to filder the paths is ignored
            dirs = [('/', d) for d in models_directories]
            # this is a tricck
            sources = []
            # ignore the following
            block = []
            where = []
            fixed_over = []
            group_by = []

        else:

            if isinstance(dataset, list):
                dataset_l = dataset
            else:
                dataset_l = [dataset]

            if not isinstance(logs_directory, list):
                logs_directory = [logs_directory]

            sources = [l + "/" + d for l in logs_directory for d in dataset_l]

            dirs = []
            for source in sources:
                print("Looking in: " + source)
                for (parent, subdirs, files) in os.walk(source):
                    # I need to remove from the list the source dir, since it is retuned by os.walk 
                    if source != parent and len(join_dirs_path(source, parent).split('/')) == depth:
                        dirs.append((source, parent))

        folders_where = [(source, d) for (source, d) in dirs if
                         check_where(join_dirs_path(source, d), where) and check_block(join_dirs_path(source, d),
                                                                                       block)]

        fixed_over_values = collections.OrderedDict()
        for f in fixed_over:
            fixed_over_values[f] = []

        if len(group_by) > 1:
            raise ValueError("The group_by only works for 0 or 1 feature, you have more: {}".format(group_by))

        group_by_values = collections.OrderedDict()
        for f in group_by:
            group_by_values[f] = []

        for (source, d) in folders_where:

            directory = join_dirs_path(source, d)
            for f_tuple in fixed_over:
                if not isinstance(f_tuple, tuple):
                    f = f_tuple
                    m = get_feature(f, directory)
                    if m:
                        if m.group(2) not in fixed_over_values[f]:
                            # here I may something like -cELBO_A1_A2-
                            # and I want to keep only ELBO
                            token = m.group(2).split("_")[0]
                            if token not in fixed_over_values[f]:
                                fixed_over_values[f].append(token)
                else:
                    (f, i) = f_tuple
                    if isinstance(i, int):
                        # i is a number and thus it specify the level in which I have to look for
                        splits = directory.split('/')
                        m = re.search('(-|^)' + f + '([\,\._A-Za-z0-9\+]+)' + '(-|$)', splits[i])
                        if m:
                            if m.group(2) not in fixed_over_values[(f, i)]:
                                fixed_over_values[(f, i)].append(m.group(2))
                    else:
                        # i is a tag, I hadve to look for the value of the sub_tag i
                        # inside the tag f
                        # pdb.set_trace()
                        m = re.search(
                            '(-|^)' + f + '([\,\._A-Za-z0-9\+]*)' + i + '([\,\.A-Za-z0-9\+]*)' + '([\,\._A-Za-z0-9\+]*)' + '(-|$)',
                            directory)
                        if m and m.group(3) not in fixed_over_values[(f, i)]:
                            token = m.group(3)
                            fixed_over_values[(f, i)].append(token)

            for f_tuple in group_by:
                if not isinstance(f_tuple, tuple):
                    f = f_tuple
                    m = get_feature(f, directory)
                    if m:
                        if m.group(2) not in group_by_values[f]:
                            group_by_values[f].append(m.group(2))
                else:
                    (f, i) = f_tuple
                    splits = directory.split('/')
                    m = get_feature(f, splits[i])
                    if m:
                        if m.group(2) not in group_by_values[(f, i)]:
                            group_by_values[(f, i)].append(m.group(2))

        print(fixed_over_values)
        print(group_by_values)

        def my_items(i):
            # return [(a[0][0], a[1], a[0][1]) if len(a) == 2 else a for a in i.items()]
            return [(a[0][0], a[1], a[0][1]) if (len(a) == 2 and isinstance(a, list)) else a for a in i.items()]

        cartesian_fixed = [collections.OrderedDict(zip(fixed_over_values.keys(), p)) for p in
                           product(*map(make_list, fixed_over_values.values()))]
        cartesian_fixed_list = [d for d in cartesian_fixed]

        for c_fixed in cartesian_fixed_list:

            folders_cartesian_fixed = [(source, d) for (source, d) in folders_where if
                                       check_where(join_dirs_path(source, d), my_items(c_fixed))]

            print("============================================")
            print(c_fixed)
            for k in folders_cartesian_fixed:
                print(k)

            n_rows, n_columns = self.rows_columns(panels, folders_cartesian_fixed)

            def format_attr(k):
                if not isinstance(k, list):
                    return str(k)
                else:
                    if isinstance(k[1], int):
                        return str(k[0]) + '[' + str(k[1]) + ']'
                    else:
                        return str(k[0]) + '_' + str(k[1])

            # needed for the filename
            attr = "-".join([format_attr(k) + str(v) for k, v in c_fixed.items()])
            if len(attr) > 0:
                attr = '_' + attr

            if isinstance(dataset, list):
                save_filename = dirname + "/" + "multiple-datasets" + "_" + filename + "_" + tag + attr
            else:
                save_filename = dirname + "/" + dataset + "_" + filename + "_" + tag + attr

            # create resource (e.g., figure or html page)
            resource = self.create_resource(figsize, folders_cartesian_fixed, suptitle)

            c = 0

            for vertical_panels in panels:
                resource = self.plot_vertical_panel(vertical_panels,
                                                    c,
                                                    n_rows,
                                                    n_columns,
                                                    legend,
                                                    resource,
                                                    folders_cartesian_fixed,
                                                    group_by_values,
                                                    degree,
                                                    string_to_val)
                c += 1

            plt.tight_layout()
            self.save_resource(resource, save_filename)
            # plt.subplots_adjust(wspace=0, hspace=0)

    def create_label(self, directory, legend, source, replacements=None):

        # check for the existence of a label.txt file
        file_path = directory + '/label.txt'
        label_decomposed = ()

        if self._labels_from_files and os.path.isfile(file_path):
            f = open(file_path, "r")
            label = f.read()
            if label[-1] == '\n':
                label = label[:-1]
            label_decomposed = label_decomposed + (label,)
        elif len(legend) > 0:
            "Legend has to be in the format: [(param, name)], where param is an id in the name"
            label = ""
            for l in legend:
                name, piece = get_value(replace_scientific(join_dirs_path(source, directory)), l)
                label_decomposed = label_decomposed + (l, self._get_true_type(piece))
                label += name + piece + ""

            label = replace_names(label, replacements)
        else:
            # label = None
            label = join_dirs_path(source, directory).replace('/', '\n')
            label_decomposed = label_decomposed + (label,)

        return label, label_decomposed

    def _get_true_type(self, piece):
        try:
            new_piece = int(piece)
        except ValueError:
            try:
                new_piece = float(piece)
            except ValueError:
                new_piece = str(piece)
        return new_piece

    def rows_columns(self, panels, folders_cartesian_fixed):
        return 0, 0

    def get_colors_pairs(self, folders_cartesian_fixed, group_by_values):
        if len(group_by_values) == 0:
            return [(i, i, d) for i, d in enumerate(folders_cartesian_fixed)]
        else:
            c_list = []
            for i, c_group in enumerate(list(group_by_values.values())[0]):
                group = [(i, (source, d)) for (source, d) in folders_cartesian_fixed if
                         check_where(join_dirs_path(source, d), [(list(group_by_values.keys())[0], c_group)])]
                c_list += [(i, *g) for i, g in enumerate(group)]

            return c_list

    @abstractmethod
    def plot_vertical_panel(self,
                            vertical_panels,
                            c,
                            n_rows,
                            n_columns,
                            legend,
                            resource,
                            folders_cartesian_fixed,
                            group_by_values,
                            degree,
                            string_to_val):
        pass

    @abstractmethod
    def create_resource(self, figsize, folders_cartesian_fixed, title):
        pass

    @abstractmethod
    def save_resource(self, resource, filename):
        pass


def check_where(s, where):
    r = True
    for tuple_where in where:
        if len(tuple_where) == 2:
            (k, list_values) = tuple_where
            if not isinstance(list_values, list):
                list_values = [list_values]
            c = False
            for v in list_values:
                if isinstance(k, tuple):
                    # match = re.search('(-|^)' + k + str(v) + '(_|-|$)', s)
                    # here I may have something like c A and 1
                    # and I need to find it in -cELBO_A1_A2-
                    match = re.search('(-|_|/|^)' + k[0] + '([\,\._A-Za-z0-9,\+]+)' + k[1] + str(v) + '(_|-|/|$)', s)
                    c = c or match
                else:
                    # match = re.search('(-|^)' + k + str(v) + '(-|$)', s)
                    # here I may have something like c and ELBO
                    # and I need to find it in -cELBO_A1_A2-
                    match = re.search('(-|_|/|^)' + k + str(v) + '(_|-|/|$)', s)
                    c = c or match
            r = r and c
        else:
            (k, list_values, i) = tuple_where
            split = s.split('/')
            if not isinstance(list_values, list):
                list_values = [list_values]
            c = False
            for v in list_values:
                match = get_specific_feature(k, v, split[i])
                c = c or match
            r = r and c

    return r


def get_specific_feature(key, val, sstr):
    return re.search('(-|_|^)' + key + str(val) + '(-|_|$)', sstr)


def check_block(s, block):
    for tuple_block in block:
        if len(tuple_block) == 2:
            (k, v) = tuple_block
            match = re.search(k + str(v) + '(-|_|$)', s)
            if match:  # k + str(v) + '-' in s or k + str(v) + '_' in s:
                return False
        else:
            (k, v, i) = tuple_block
            split = s.split('/')
            match = re.search(k + str(v) + '(-|_|$)', split[i])
            if match:  # k + str(v) + '-' in split[i] or k + str(v) + '_' in split[i]:
                return False
    return True


def join_dirs_path(source, d):
    return d.replace(source + '/', '').replace('//', '/')  # .replace('/','-')


def replace_scientific(string):
    # scientific notation numbers
    snlist = re.findall('(\d+[eE]([+-])(\d+))', string)
    if len(snlist) > 0:
        def process_sn(number, sign, exp):
            if sign == "-":
                stringy = ('{:.' + str(int(exp)) + 'f}').format(float(number))
            else:
                stringy = ('{:' + str(int(exp)) + 'd}').format(int(float(number)))
            return (number, stringy)

        sncouples = [process_sn(*sn) for sn in snlist]

        newstring = copy.deepcopy(string)
        for oldstr, newstr in sncouples:
            newstring = newstring.replace(oldstr, newstr)

        return newstring
    return string


def replace_names(label, replacements):
    if replacements is not None:
        for original, replace in replacements:
            label = label.replace(original, replace, 1)
    return label


def get_field(string, field_spec):
    l = len(field_spec)

    if l == 0 or l > 2:
        raise ValueError("Not implemented tuple length `{:}`, found field spec `{:}`".format(l, field_spec))

    m = re.search('(-|^)' + field_spec[0] + '([\._A-Za-z0-9\,]+)' + '(-|$)', string)
    if m is None:
        ss1 = '0'
    else:
        ss1 = m.group(2)

    if l == 1:
        return ss1

    m = re.search('(_|^)' + field_spec[1] + '([\.A-Za-z0-9\,]+)' + '(_|$)', ss1)

    if m is None:
        ss2 = '0'
    else:
        ss2 = m.group(2)

    return ss2


def get_value(directory, l):
    if len(l) == 2:
        (param, name) = l
        m = re.search('(-|^)' + param + '([\,\._A-Za-z0-9\+]+)' + '(-|/|$)', directory)
        if m and m.group(2):
            return name, m.group(2)
        return "", ""  # "param not found"
    elif len(l) == 3:
        (param, i, name) = l
        splits = directory.split('/')
        # pdb.set_trace()
        m = re.search('(-|^)' + param + '([\,\._A-Za-z0-9\+]+)' + '(-|/|$)', splits[i])
        if m and m.group(2):
            return name, m.group(2)
        return "", ""  # "param not found"
    elif len(l) == 4:
        (param, subparam, i, name) = l
        splits = directory.split('/')
        m = re.search('(-|^)' + param + '([\,\._A-Za-z0-9\+]+)' + '(-|/|$)', splits[i])
        if subparam == "":  # in case you want just the param without the rest
            if m and m.group(2):
                m = re.search('(-|^)' + param + '([\,\.A-Za-z0-9\+]+)' + '(_|/|$)', splits[i])
                if m and m.group(2):
                    return name, m.group(2)
        if m and m.group(2):
            found = m.group(2)
            m = re.search('(_)' + subparam + '([\,\.A-Za-z0-9\+]+)' + '(_|/|$)', found)
            if m and m.group(2):
                return name, m.group(2)
        return "", ""  # "param not found"
    else:
        raise Exception("The length of the token should be 2 or 3")

    return "param not found"


def get_feature(feature, directory):
    return re.search('(-|_|/|^)' + feature + '([\,\.A-Za-z0-9\+]+)' + '(-|_|/|$)', directory)
