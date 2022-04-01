# it takes a lifetime to import this (****)
import matplotlib
from matplotlib import colors as mcolors

from core.argo.core.utils.Curve import Curve
from .AbstractPlot import AbstractPlot

import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

# see https://matplotlib.org/api/markers_api.html
markers = "ov^<>spP*Dd"  # +x

def fill_missing_values_in_legend(labels_and_folders_cartesian_fixed):
    np_labels_and_folders_cartesian_fixed = np.asarray([np.asarray(list(tup)) for tup in labels_and_folders_cartesian_fixed])
    labels = np_labels_and_folders_cartesian_fixed[:, 0]
    labels = pd.DataFrame(np.asarray(list(map(lambda x: list(x), labels))))
    for col in labels.columns:
        label_list = labels[col]
        if np.all([label_list[0] == lab for lab in label_list]):
            continue

        #is there a float
        if np.any([type(lab) == float for lab in label_list]):
            label_list[[type(lab) != float for lab in label_list]] = 0.0

        # is there an int
        elif np.any([type(lab) == int for lab in label_list]):
            label_list[[type(lab) != int for lab in label_list]] = 0


    np_labels_and_folders_cartesian_fixed[:, 0] = list([tuple(lab) for lab in labels.to_numpy()])

    return [tuple(lab) for lab in list(np_labels_and_folders_cartesian_fixed)]

class PlotMatplotlibCurves(AbstractPlot):

    def __init__(self):
        super(PlotMatplotlibCurves, self).__init__()

    def create_resource(self, figsize, folders_cartesian_fixed, title):
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, y=0.995)
        return fig

    # could be moved in a AbstactMatplotlibPlot
    def save_resource(self, resource, filename):
        print("saving: " + filename + ".png")
        resource.savefig(filename + ".png", bbox_inches='tight')

    def rows_columns(self, panels, folders_cartesian_fixed):
        n_columns = len(panels)
        n_rows = np.max([len(p) for p in panels])
        return n_rows, n_columns

    def plot_vertical_panel(self,
                            vertical_panels,
                            c,
                            n_rows,
                            n_columns,
                            legend,
                            fig,
                            folders_cartesian_fixed,
                            group_by_values,
                            degree,
                            string_to_val):

        r = 0

        for panel in vertical_panels:

            ax = fig.add_subplot(n_rows, n_columns, r * n_columns + c + 1)

            if len(group_by_values) == 0:
                max_colors = len(folders_cartesian_fixed)
            else:
                max_colors = len(list(group_by_values.values())[0])

            if max_colors == 0:
                max_colors = 1
            list_colors = self.create_list_colors(max_colors)

            firstCurve = None

            for curve_spec in panel:
                # This whole thing is made to sort the labels by their data type. Before this, all labels were sorted as
                # str, so the order was sometimes 1,10,100,2,20,5,50,500, which is confusing
                labels_and_folders_cartesian_fixed = [
                    (*self.create_label(directory, legend, source, self._replace_names)[::-1], source, directory) for
                    (source, directory) in folders_cartesian_fixed]

                labels_and_folders_cartesian_fixed = fill_missing_values_in_legend(labels_and_folders_cartesian_fixed)

                folders_cartesian_fixed_sorted_label = sorted(labels_and_folders_cartesian_fixed)

                folders_cartesian_fixed_sorted = list(zip(*list(zip(*folders_cartesian_fixed_sorted_label))[-2:]))

                labels_sorted = list(zip(*folders_cartesian_fixed_sorted_label))[1] if len(folders_cartesian_fixed_sorted) > 0 else []

                marker_color_folders_cartesian_fixed_sorted = [ (marker, color, source, directory) for
                    marker, color, (source, directory) in self.get_colors_pairs(folders_cartesian_fixed_sorted,
                                                                      group_by_values)]


                for label, (marker, color, source, directory) in zip(labels_sorted, marker_color_folders_cartesian_fixed_sorted):
                    if not "legend_loc" in curve_spec:
                        label=None
                    cur = Curve(directory=directory, string_to_val=string_to_val, label=label, **curve_spec)

                    if firstCurve is None:
                        firstCurve = cur

                    cur.create_plot(markers[marker % len(markers)], list_colors[color], ax, degree)

            # sometimes it raises an error, so I added a check (****)
            if firstCurve is not None:
                firstCurve.set_panel_properties(ax)

            r += 1

        return fig

    def create_list_colors(self, n):
        if len(self._colors) == 0:
            if n <= 10:
                colors = list(mcolors.TABLEAU_COLORS.values())
            elif n > 10 and n <= 20:
                colors = list(matplotlib.cm.get_cmap("tab20").colors)
            else:
                cmap = matplotlib.cm.get_cmap("gist_rainbow", n)
                colors = [cmap(i) for i in range(n)]

            return [mcolors.to_hex(rgba) for rgba in colors]
        else:
            if len(self._colors) >= n:
                return [mcolors.to_hex(rgba) for rgba in self._colors]
            else:
                raise Exception("Number of plots ({}) exceeds number of colors ({}): {}".format(n, len(self._colors), self._colors))
