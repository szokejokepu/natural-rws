import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator


def str2float(s, string_to_val):
    try:
        val = string_to_val[s]
    except:
        val = float(s)

    return val


class Curve():

    def __init__(self, directory,
                 string_to_val,
                 label,
                 filename,
                 x,
                 y,
                 logscale_x=0,
                 logscale_y=0,
                 accumulate_x=0,
                 accumulate_y=0,
                 smoothing=0,
                 x_label="",
                 y_label=None,
                 separator="\t",
                 limit=0,  # If you want just a limited number of elements from your x and y
                 min_y=None,
                 min_x=None,
                 max_y=None,
                 max_x=None,
                 scale_x=1.0,
                 scale_y=1.0,
                 x_ticks=None,
                 y_ticks=None,
                 mark_every=25,
                 linewidth=2,
                 linestyle='-',
                 legend_loc="upper right",
                 bbox_to_anchor=None,
                 legend_ncol=1,
                 ticks=None,
                 legend=True,
                 prefix="",
                 median_weighting_x=False,
                 median_weighting_y=False):
        self.directory = directory
        self.string_to_val = string_to_val
        self.label = label
        self.filename = filename
        self.logscale_x = logscale_x
        self.logscale_y = logscale_y
        self.accumulate_x = accumulate_x
        self.accumulate_y = accumulate_y
        self.smoothing = smoothing
        self.x_label = x_label
        self.y_label = y_label
        self.separator = separator
        self.min_y = min_y
        self.min_x = min_x
        self.max_y = max_y
        self.max_x = max_x
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.xticks = x_ticks
        self.yticks = y_ticks
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.legend_loc = legend_loc
        self.bbox_to_anchor = bbox_to_anchor
        self.legend_ncol = legend_ncol
        self.ticks = ticks
        self.legend = legend
        self.prefix = prefix
        self.median_weighting_x = median_weighting_x
        self.median_weighting_y = median_weighting_y

        assert self.filename is not None, "Filename Can't be empty"
        file_path = self.directory + "/" + self.filename

        self._plotable = False

        if isinstance(x, int) and isinstance(y, int):
            x_read, y_read = self.read_manually_from_file(file_path, x, y)
        elif isinstance(x, str) and isinstance(y, str):
            x_read, y_read = self.read_with_pandas(file_path, x, y)
        else:
            raise Exception("Both x and y should be either ints or str as column names")
        if limit != 0:
            x_read = x_read[0:limit + 1]
            y_read = y_read[0:limit + 1]

        self.x = self.scale_x * np.array(x_read)
        self.y = self.scale_y * np.array(y_read)

        if np.asarray(self.y).size > 1:
            self.mark_every = min(len(self.y), mark_every)  # int(len(y) / 20)
            self._plotable = True
        else:
            print("Not enough data to plot {}/{}".format(self.directory, self.filename))
            return

        if self.accumulate_x == 1:
            if self.median_weighting_x:
                median_x = np.median(self.x)
                self.x = median_x * np.ones([len(self.x)])
            self.x = np.add.accumulate(self.x)

        if self.accumulate_y == 1:
            if self.median_weighting_y:
                median_y = np.median(self.y)
                self.y = median_y * np.ones([len(self.y)])
            self.y = np.add.accumulate(self.y)

    def create_plot(self, marker, color, ax, degree):

        if self._plotable:

            if self.logscale_x != 0:
                ax.set_xscale('log')

            if self.logscale_y != 0:
                plot_function = ax.semilogy
            else:
                plot_function = ax.plot
            label = self.prefix + " " + self.label if self.label is not None else None
            if self.smoothing != 0 and len(self.y) > 60:
                # see https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
                y_smooth = smoothTriangle(self.y, degree)  # window size 51, polynomial order 3
                plot_function(self.x, y_smooth, self.linestyle, c=color,
                              label=label, marker=marker, markevery=self.mark_every,
                              markersize=10, linewidth=self.linewidth)
                plot_function(self.x, self.y, self.linestyle, c=color, alpha=0.0,
                              linewidth=self.linewidth)
            else:
                plot_function(self.x, self.y, self.linestyle, c=color, label=label,
                              marker=marker, markevery=self.mark_every, markersize=10,
                              linewidth=self.linewidth)

    def set_panel_properties(self, ax):
        # first curve of the panel

        ax.set_xlim(left=self.min_x)
        ax.set_xlim(right=self.max_x)

        ax.set_ylim(bottom=self.min_y)
        ax.set_ylim(top=self.max_y)

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)

        if self.xticks is not None:
            ax.set_xticks(self.xticks)
        if self.yticks is not None:
            ax.set_yticks(self.yticks)

        # Options for legend_loc
        # 'best', 'upper right', 'upper left', 'lower left', 'lower right',
        # 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
        if self.legend:
            if self.bbox_to_anchor is None:
                lgd = ax.legend(loc=self.legend_loc, ncol=self.legend_ncol)
            else:
                lgd = ax.legend(loc=self.legend_loc, bbox_to_anchor=self.bbox_to_anchor, ncol=self.legend_ncol)

        if self.ticks is not None:
            xticks_spec = self.ticks['x']
            xstep = xticks_spec.get('step', 1)
            xceil = xticks_spec.get('ceil', 1)
            xnminor = xticks_spec.get('nminor', 0)
            ax.set_xticks(np.arange(self.min_x, self.max_x + xceil, xstep))
            minor_locator = AutoMinorLocator(xnminor)
            ax.xaxis.set_minor_locator(minor_locator)

            yticks_spec = self.ticks['y']
            ystep = yticks_spec.get('step', 1)
            yceil = yticks_spec.get('ceil', 1)
            ynminor = yticks_spec.get('nminor', 0)
            ax.set_yticks(np.arange(self.min_y, self.max_y + yceil, ystep))
            minor_locator = AutoMinorLocator(ynminor)
            ax.yaxis.set_minor_locator(minor_locator)

            ax.set_axisbelow(True)

            major_lines = self.ticks.get("major", {})
            minor_lines = self.ticks.get("minor", {})

            lnst = major_lines.get('linestyle', '--')
            lnw = major_lines.get('linewidth', 1)
            lnc = major_lines.get('color', 'lightgray')
            ax.grid(which='major', linestyle=lnst, linewidth=lnw, color=lnc)

            lnst = minor_lines.get('linestyle', '--')
            lnw = minor_lines.get('linewidth', 1)
            lnc = minor_lines.get('color', 'lightgray')
            ax.grid(which='minor', linestyle=lnst, linewidth=lnw, color=lnc)

        else:
            ax.grid()

    def read_with_pandas(self, file_path, x, y):
        df = pd.read_csv(file_path, sep=self.separator)
        df.columns = [i.strip() for i in df.columns]
        # import ipdb; ipdb.set_trace()
        assert x in df.columns, "No such column \"{}\" in columns: {}".format(x, df.columns)
        assert y in df.columns, "No such column \"{}\" in columns: {}".format(y, df.columns)
        return df[x].to_numpy(), df[y].to_numpy()

    def read_manually_from_file(self, file_path, x, y):
        try:
            # read data from file
            with open(file_path) as f:
                data = f.read()

            data = data.split('\n')

            offset = 1
            data = data[offset:-1]

            separator = self.separator
            x = [str2float(row.lstrip().split(separator)[x], self.string_to_val) for row in data]
            y = [str2float(row.lstrip().split(separator)[y], self.string_to_val) for row in data]

        except FileNotFoundError as e:
            print("FileNotFoundError cannot read file " + file_path + " " + str(e))
        except AssertionError as e:
            print("AssertionError file is empty " + file_path + " " + str(e))
        except ValueError as e:
            print("ValueError cannot read file " + file_path + " " + str(e))
        except IndexError as e:
            print("IndexError cannot read file " + file_path + " " + str(e))

        return x, y


# see https://www.swharden.com/wp/2010-06-20-smoothing-window-data-averaging-in-python-moving-triangle-tecnique/
def smoothTriangle(data, degree, dropVals=False):
    """performs moving triangle smoothing with a variable degree."""
    """note that if dropVals is False, output length will be identical
    to input length, but with copies of data at the flanking regions"""
    triangle = np.array(list(range(degree)) + [degree] + list(range(degree))[::-1]) + 1
    smoothed = []
    for i in range(degree, len(data) - degree * 2):
        point = data[i:i + len(triangle)] * triangle
        smoothed.append(sum(point) / sum(triangle))
    if dropVals: return smoothed
    smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed
    while len(smoothed) < len(data): smoothed.append(smoothed[-1])

    # some refinement
    alphas = [alpha / degree / 2 for alpha in range(degree * 2)]
    smoothed[:degree * 2] = [a * s + (1 - a) * d for a, s, d in zip(alphas, smoothed[:degree * 2], data[:degree * 2])]
    return smoothed
