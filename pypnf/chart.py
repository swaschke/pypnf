# -*- coding: utf-8 -*-
#
# pyPnF
# A Package for Point and Figure Charting
# https://github.com/swaschke/pypnf
#
# Copyright (C) 2021  Stefan Waschke
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
The chart module of the pyPnF package contains the main class PointFigureChart.
The class handles the chart parameter and contains basic attributes like breakouts and trendlines.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tabulate import tabulate
from warnings import warn


class PointFigureChart:
    """ Class to build a Point and Figure Chart from time series data

    Required attributes:
    ====================

    ts: dict
        with keys ['date','open','high','low','close',volume']
    ts['date']:
        Optional. Array or value of type str
    ts['open']:
        Array or value of type float
    ts['high']:
        Array or value of type float
    ts['low']:
        Array or value of type float
    ts['close']:
        Array or value of type float
    ts['volume']:
        Optional. array or value of type int/float

    :method: str
        methods implemented: 'cl', 'h/l', 'l/h', 'hlc', 'ohlc' default('cl')
    boxscaling: str
        scales implemented:
            'abs', 'atr', 'cla', 'log' default('log')
        abs:
            absolute scaling with fixed box sizes.
        atr:
            absolute scaling with atr of last n periods
        log:
            logarithmic scaling with variable box sizes.
        cla:
            classic scaling with semi-variable box sizes.
    boxsize: int/float/string
        Size of boxes with regards to the respective scaling default (1).
        Implemented box sizes for classic scaling are 0.02, 0.05, 0.1, 0.25, 1/3, 0.5, 1, 2.
        For classic scaling the box size serves as factor to scale the original scale.
        The minimum boxsize for logarithmic scaling is 0.01%.
        For atr scaling the number of last n periods to calculate from, 'total' for all periods.
    title: str
        user defined label for the chart default(None)
        label will be created inside the class.
        The label contains contains the chart parameter and the title.

    Methods:
    ========
    get_breakouts(): dict
        Gets breakout points for Point and Figure Charts.
        Detailed description in get_breakouts-method.
    get_trendlines(length, mode): dict
        Gets trendlines for Point and Figure Charts.
        Detailed description in get_trendlines-method.


    Returned attributes:
    ====================

    pnf_timeseries: dict
        pnf_timeseries['date']: str or int
            Array or value of type str if datetime
        pnf_timeseries['box value']: float
            Array with prices of the last filled box
        pnf_timeseries['box index']: float
            Array with indices of the last filled box.
        pnf_timeseries['column index']: float
            Array with indices of the current column.
        pnf_timeseries['trend']: float
            Array with values for the current trend.
            Uptrends:    1
            Downtrends: -1
        pnf_timeseries['filled boxes']: float
            Array with values for number of filled boxes in the current column.

        Note:
            Due to the usage of numpy.nan all indices are of type float instead of int.

    boxscale: numpy.ndarray
        1-dim numpy array with box values in increasing order.
    matrix: numpy.ndarray
        2-dim numpy array representing the Point and Figure Chart
        with values 0, 1 and -1. Zero represents an unfilled box,
        One a box filled with an X and neg. One filled with an O.
        Columns are equivalent to the chart columns, rows to the
        corresponding index in the boxscale.
    trendlines: dict
       Detailed description in get_trendline-method.
    title: str
        Label containing chart parameter and user-defined title.
    breakouts: dict
        Detailed description in get_breakouts-method.
    """

    def __init__(self, ts, method='cl', reversal=3, boxsize=1, scaling='log', title=None):

        # chart parameter
        self.method = self._is_valid_method(method)
        self.reversal = self._is_valid_reversal(reversal)
        self.scaling = self._is_valid_scaling(scaling)
        self.boxsize = self._is_valid_boxsize(boxsize)

        # prepare timeseries
        self.time_step = None  # to be calculated in _prepare_ts
        self.ts = self._prepare_ts(ts)

        # chart
        self.title = self._make_title(title)
        self.boxscale = self._get_boxscale()
        self.pnf_timeseries = self._get_pnf_timeseries()
        self.matrix = self._pnf_timeseries2matrix()
        self.column_labels = self._get_column_entry_dates()

        # trendlines
        self.trendlines = None
        self.show_trendlines = False  # 'external', 'internal', 'both', False, 'False'

        # signals
        self.breakouts = None
        self.buys = {}
        self.sells = {}
        self.show_breakouts = False
        self.bullish_breakout_color = 'g'
        self.bearish_breakout_color = 'm'

        # indicator
        self.column_midpoints = None
        self.indicator = {}
        self.vap = {}
        self.indicator_colors = plt.cm.Set2
        self.indicator_fillcolor_opacity = 0.2

        # plotting coordinates/adjusted indicator
        self.plot_boxscale = None
        self.plot_matrix = None
        self.plot_column_index = None
        self.plot_column_label = None
        self.plot_y_ticks = None
        self.plot_y_ticklabels = None
        self.matrix_top_cut_index = None
        self.matrix_bottom_cut_index = None
        self.plot_indicator = {}
        self.cut2indicator = False
        self.cut2indicator_length = None

        # plotting options
        self.size = 'auto'
        self.max_figure_width = 10
        self.max_figure_height = 8
        self.left_axis = False
        self.right_axis = True
        self.column_axis = True

        self.add_empty_columns = 0

        self.show_markers = True
        self.grid = None
        self.x_marker_color = 'grey'
        self.o_marker_color = 'grey'
        self.grid_color = 'grey'

        self.figure_width = None
        self.figure_height = None
        self.matrix_min_width = None

        self.margin_left = None
        self.margin_right = None
        self.margin_top = 0.3
        self.margin_bottom = None
        self.box_height = None

        self.marker_linewidth = None
        self.grid_linewidth = None

        self.x_label_step = None
        self.y_label_step = None

        self.label_fontsize = 8
        self.title_fontsize = 8
        self.legend_fontsize = 8

        self.legend = True
        self.legend_position = None
        self.legend_entries = None

        self.plotsize_options = {'size': ['huge', 'large', 'medium', 'small', 'tiny'],
                                 'grid': [True, True, True, False, False],
                                 'matrix_min_width': [12, 12, 27, 57, 117],
                                 'box_height': [0.2, 0.15, 0.1, 0.05, 0.025],
                                 'marker_linewidth': [1, 1, 1, 0.5, 0.5],
                                 'grid_linewidth': [0.5, 0.5, 0.5, 0.25, 0.125],
                                 'x_label_step': [1, 1, 2, 4, 8],
                                 'y_label_step': [1, 1, 2, 4, 8],
                                 }

        # Figure and axis objects
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None

    @staticmethod
    def _is_valid_method(method):

        if method not in ['cl', 'h/l', 'l/h', 'hlc', 'ohlc']:
            raise ValueError("Not a valid method. Valid methods are: cl, h/l, l/h, hlc, ohlc")

        return method

    @staticmethod
    def _is_valid_reversal(reversal):

        if not isinstance(reversal, int):
            ValueError('Value for reversal must be an integer. Reversal is usually between 1 and 5.')

        return reversal

    @staticmethod
    def _is_valid_scaling(scaling):

        if scaling not in ['abs', 'log', 'cla', 'atr']:
            raise ValueError("Not a valid scaling. Valid scales are: abs, log, cla and atr")

        return scaling

    def _is_valid_boxsize(self, boxsize):

        if self.scaling == 'cla':

            valid_boxsize = [0.02, 0.05, 0.1, 0.25, 1 / 3, 0.5, 1, 2]

            if boxsize not in valid_boxsize:
                msg = 'ValueError: For cla scaling valid values for boxsize are 0.02, 0.05, 0.1, 0.25, 1/3, 0.5, 1, 2'
                raise ValueError(msg)

        elif self.scaling == 'log':
            if boxsize < 0.01:
                raise ValueError('ValueError: The smallest possible boxsize for log-scaled axis is 0.01%')

        elif self.scaling == 'abs':
            if boxsize < 0:
                raise ValueError('ValueError: The boxsize must be a value greater than 0.')
                
        elif self.scaling == 'atr':
            if boxsize != 'total' and int(boxsize) != boxsize:
                raise ValueError('ValueError: The boxsize must be a integer of periods or \'total\' for atr box scaling.')
                
            if boxsize != 'total' and boxsize < 0:
                raise ValueError('ValueError: The boxsize must be a value greater than 0.')
            

            
        return boxsize

    def _make_title(self, title):

        if title is None:

            if self.scaling == 'log':
                title = f'Point & Figure ({self.scaling}|{self.method}) {self.boxsize}% x {self.reversal}'

            elif self.scaling == 'cla':
                title = f'Point & Figure ({self.scaling}|{self.method}) {self.boxsize}@50 x {self.reversal}'

            elif self.scaling == 'abs' or self.scaling == 'atr':
                title = f'Point & Figure ({self.scaling}|{self.method}) {self.boxsize} x {self.reversal}'

        else:

            if self.scaling == 'log':
                title = f'Point & Figure ({self.scaling}|{self.method}) {self.boxsize}% x {self.reversal} | {title}'

            elif self.scaling == 'cla':
                title = f'Point & Figure ({self.scaling}|{self.method}) {self.boxsize}@50 x {self.reversal} | {title}'

            elif self.scaling == 'abs' or self.scaling == 'atr':
                title = f'Point & Figure ({self.scaling}|{self.method}) {self.boxsize} x {self.reversal} | {title}'

        return title

    def _prepare_ts(self, ts):
        """
        Initiates the time series data and adjust to the required format.
        """

        # bring all keys to lowercase characters
        ts = {key.lower(): val for key, val in ts.items()}
        
        # check if all required keys are available
        if self.method == 'cl':

            if 'close' not in ts:
                raise KeyError("The required key 'close' was not found in ts")

        elif self.method == 'h/l' or self.method == 'l/h':

            if 'low' not in ts:
                raise KeyError("The required key 'low' was not found in ts")

            if 'high' not in ts:
                raise KeyError("The required key 'high' was not found in ts")

        elif self.method == 'hlc':

            if 'close' not in ts:
                raise KeyError("The required key 'close' was not found in ts")

            if 'low' not in ts:
                raise KeyError("The required key 'low' was not found in ts")

            if 'high' not in ts:
                raise KeyError("The required key 'high' was not found in ts")

        elif self.method == 'ohlc' or self.scaling == 'atr':

            if 'close' not in ts:
                raise KeyError("The required key 'close' was not found in ts")

            if 'low' not in ts:
                raise KeyError("The required key 'low' was not found in ts")

            if 'high' not in ts:
                raise KeyError("The required key 'high' was not found in ts")

            if 'open' not in ts:
                raise KeyError("The required key 'open' was not found in ts")
                
        if self.scaling == 'atr':
            if self.boxsize != 'total' and self.boxsize + 1 > len(ts['close']):
                raise IndexError("ATR boxsize is larger than length of data.")
                
        # bring all inputs to the final format as dict with numpy.ndarrays.
        for key in ts.keys():
            if isinstance(ts[key], list):
                ts[key] = np.array(ts[key])
            if not type(ts[key]) == np.ndarray:
                if type(ts[key]) == str or float or int:
                    ts[key] = np.array([ts[key]])

                    # if ts['date'] exist check for the type, if it's a string convert
        # to datetime64 else create index of integers.
        # If the string can't converted to datetime64 create index of integers.
        if 'date' not in ts:
            ts['date'] = np.arange(0, ts['close'].shape[0])

        if isinstance(ts['date'][0], str):

            try:
                ts['date'] = ts['date'].astype('datetime64')

                datetime_diff = ts['date'][0:-1] - ts['date'][1:]

                if any(np.mod(datetime_diff / np.timedelta64(1, "D"), 1) != 0):
                    self.time_step = 'm'
                elif any(np.mod(datetime_diff / np.timedelta64(1, "D"), 1) == 0):
                    self.time_step = 'D'
                else:
                    self.time_step = None

            except ValueError:
                warn('Date string can`t be converted to datetime64. Date is set to index of integers')
                ts['date'] = np.arange(0, ts['close'].shape[0])

        # if date is datetime64 check if last date in array is the latest and
        # flip the array if not.
        if isinstance(ts['date'][0], np.datetime64):
            if ts['date'][0] > ts['date'][-1]:
                for key in ts.keys():
                    ts[key] = np.flip(ts[key])

        if not isinstance(ts['date'][0], np.datetime64):
            ts['date'] = np.arange(0, ts['close'].shape[0])

        # check if all arrays have the same length
        length = [x.shape[0] for x in ts.values()]
        if not all(x == length[0] for x in length):
            raise IOError('All arrays in the time-series must have the same length')

        return ts

    def _get_boxscale(self, overscan=None):
        """
        creates the box scale for Point and Figure Chart
        """

        if self.method == 'cl':
            minimum = np.min(self.ts['close'])
            maximum = np.max(self.ts['close'])
        else:
            minimum = np.min(self.ts['low'])
            maximum = np.max(self.ts['high'])

        # initiate variable for boxscale
        boxes = np.array([])

        # initiate overscan range for top and bottom of the scale
        overscan_top = 0
        overscan_bot = 0

        # define range for overscan. If no value is given take the reversal
        if overscan is None:
            overscan = 20  # self.reversal

        if type(overscan) == int:
            overscan_bot = overscan
            overscan_top = overscan
        elif type(overscan) == list or type(overscan) == tuple:
            overscan_bot = overscan[0]
            overscan_top = overscan[1]

        # make scale for absolute scaling
        if self.scaling == 'abs' or self.scaling == 'atr':
            if self.scaling == 'atr':
                
                # Calculate components of the True Range
                p = self.boxsize == 'total' and len(self.ts['close'])-1 or self.boxsize
                high_low = self.ts['high'][-p:] - self.ts['low'][-p:]
                high_close_prev = np.abs(self.ts['high'][-p:] - self.ts['close'][-p-1:-1])
                low_close_prev = np.abs(self.ts['low'][-p:] - self.ts['close'][-p-1:-1])
                
                # Combine and find the maximum for each day to get the True Range, excluding the first day due to shift
                true_range = np.maximum(np.maximum(high_low, high_close_prev), low_close_prev)
                
                # Calculate a single average value for the True Range, to be used as the box size
                self.boxsize = np.mean(true_range)
                
                self.scaling = 'abs'
                
            decimals = len(str(self.boxsize).split(".")[-1])

            boxes = np.array(np.float64([0]))
            boxsize = np.round(np.float64(self.boxsize), decimals)

            while boxes[0] <= minimum - (overscan_bot + 1) * boxsize:
                boxes[0] = np.round(boxes[0] + boxsize, decimals)

            n = 0
            while boxes[n] <= maximum + (overscan_top - 1) * boxsize:
                boxes = np.append(boxes, np.round(boxes[n] + boxsize, decimals))
                n += 1

        # make scale for logarithmic scaling
        elif self.scaling == 'log':

            boxsize = np.float64(self.boxsize)
            minval = 0.0001  # minimum value for log-scaled axis

            boxes = np.array([np.log(minval)])
            log_boxsize = np.log(1 + boxsize / 100)

            while boxes[0] <= np.log(minimum) - (overscan_bot + 1) * log_boxsize:
                boxes[0] = boxes[0] + log_boxsize

            n = 0
            while boxes[n] <= np.log(maximum) + (overscan_top - 1) * log_boxsize:
                boxes = np.append(boxes, boxes[n] + log_boxsize)
                n += 1

            boxes = np.exp(boxes)

            if boxsize >= 0.1:
                boxes = np.where((boxes >= 0.1) & (boxes < 1), np.round(boxes, 5), boxes)
                boxes = np.where((boxes >= 1) & (boxes < 10), np.round(boxes, 4), boxes)
                boxes = np.where((boxes >= 10) & (boxes < 100), np.round(boxes, 3), boxes)
                boxes = np.where(boxes >= 100, np.round(boxes, 2), boxes)

        # make scale for classic scaling
        elif self.scaling == 'cla':

            f = self.boxsize
            s = np.array([0.2, 0.5, 1]) * f

            b1 = np.arange(6, 14 - s[0], s[0])
            b2 = np.arange(14, 29 - s[1], s[1])
            b3 = np.arange(29, 60 - s[2], s[2])

            b0 = np.hstack((b1, b2, b3)) / 10000

            g = np.array([1])
            boxes = np.append(0, b0 * g)

            while boxes[-overscan_top - 1] < maximum:
                g = g * 10
                boxes = np.append(boxes, np.round(b0 * g, 5))

            start = np.where(boxes <= minimum)[0][-1] - overscan_bot
            if start < 0:
                start = 0
            end = np.where(boxes > maximum)[-1][0] + overscan_top

            boxes = boxes[start:end]

        return boxes

    def _get_first_trend(self):
        """
        Determines the first box and trend
        """

        if self.method == 'cl' or self.method == 'ohlc':
            H = self.ts['close']
            L = self.ts['close']
        else:
            H = self.ts['high']
            L = self.ts['low']

        Boxes = self.boxscale

        iBu = np.where(Boxes >= H[0])[0][0]

        if H[0] != Boxes[iBu]:
            iBu = iBu - 1

        iBd = np.where(Boxes <= L[0])[0][-1]

        k = 1
        uTF = 0  # uptrend flag
        dTF = 0  # downtrend flag

        while uTF == 0 and dTF == 0 and k <= np.size(H) - 1:
            if H[k] >= Boxes[iBu + 1]:
                uTF = 1
            else:
                if L[k] <= Boxes[iBd - 1]:
                    dTF = -1
            k += 1

        # first trend is up
        if uTF > 0:
            TF = uTF
            iB = iBu

        # first trend is down
        elif dTF < 0:
            TF = dTF
            iB = iBd

        # no trend
        else:
            TF = 0
            iB = 0

        iC = 0  # column index
        fB = 1  # number of filled Boxes
        box = Boxes[iB]

        iD = k - 1  # index of date with first entry

        if TF == 0:
            raise ValueError('Choose a smaller box size. There is no trend using the current parameter.')

        return iD, box, iB, iC, TF, fB

    def _basic(self, P, iB, iC, TF, fB):
        """
        basic logic to build point and figure charts
        """

        Boxes = self.boxscale
        reversal = self.reversal

        iBp = iB  # Box index from previous iteration
        fBp = fB  # number of filled Boxes from previous iteration

        if TF == 1:

            # check if there is a further 'X' in the trend
            if P >= Boxes[iB + 1]:

                # increase box index until the price reaches the next box level
                while P >= Boxes[iB + 1]:
                    iB = iB + 1

                # calculate number of filled Boxes
                fB = fB + iB - iBp

            # the Box index can not be zero
            if iB - reversal < 1:
                iB = 1 + reversal

            # check for reversal
            if P <= Boxes[iB - reversal]:

                # set Box index to the bottom box
                iB = np.where(Boxes >= P)[0][0]

                TF = -1  # trend becomes negative
                iC = iC + 1  # go to next column
                fB = iBp - iB  # calculate number of filled Boxes

                # check for one-step-back
                if reversal == 1 and fBp == 1:
                    iC = iC - 1  # set column to previous column
                    fB = fB + 1  # calculate number of filled Boxes

        elif TF == -1:

            # the Box index can not be zero
            if iB - 1 < 1:
                iB = 1 + 1

            # check if there is a further 'O' in the trend
            if P <= Boxes[iB - 1]:

                # decrease box index until the price falls down under the next box level
                while P <= Boxes[iB - 1]:
                    iB = iB - 1

                # calculate number of filled Boxes
                fB = fB + iBp - iB

            # check for reversal
            if P >= Boxes[iB + reversal]:

                # set Box index to the top box
                iB = np.where(Boxes <= P)[0][-1]

                TF = 1  # trend becomes positive
                iC = iC + 1  # go to next column
                fB = iB - iBp  # calculate number of filled Boxes

                # check for one-step-back
                if reversal == 1 and fBp == 1:
                    iC = iC - 1  # set column to previous column
                    fB = fB + 1  # calculate number of filled Boxes

        Box = Boxes[iB]

        return Box, iB, iC, TF, fB

    def _close(self, iD, Box, iB, iC, TF, fB):
        """
        logic for point and figure charts based on closing prices
        """

        C = self.ts['close']

        ts = np.zeros([np.size(C), 5])

        # make the first entry right before the first change
        # otherwise filled boxes can be not correctly determined
        # in next iteration.
        ts[0: iD, :] = [Box, iB, iC, TF, fB]

        C = C[iD:]

        for n, C in enumerate(C):
            [Box, iB, iC, TF, fB] = self._basic(C, iB, iC, TF, fB)
            ts[iD + n, :] = [Box, iB, iC, TF, fB]

        return ts

    def _hilo(self, iD, Box, iB, iC, TF, fB):
        """
        logic for point and figure charts adapting the high/low method
        """

        H = self.ts['high']
        L = self.ts['low']

        Boxes = self.boxscale
        reversal = self.reversal

        ts = np.zeros([np.size(H), 5])

        # make the first entry right before the first change
        # otherwise filled boxes can be not correctly determined
        # in next iteration.
        ts[0: iD, :] = [Box, iB, iC, TF, fB]

        for n in range(iD, np.size(H)):

            iBp = iB  # Box index from previous iteration
            fBp = fB  # number of filled Boxes from previous iteration

            if TF == 1:

                # check if there is a further 'X' in the trend
                if H[n] >= Boxes[iB + 1]:
                    [Box, iB, iC, TF, fB] = self._basic(H[n], iB, iC, TF, fB)

                else:

                    # the Box index can not be zero
                    if iB - reversal < 1:
                        iB = 1 + reversal

                    # check low for reversal
                    if L[n] <= Boxes[iB - reversal]:
                        TF = -1
                        [Box, iB, iC, TF, _] = self._basic(L[n], iB, iC, TF, fB)
                        iC = iC + 1  # go to next column
                        fB = iBp - iB  # calculate number of filled Boxes

                        # check for one-step-back
                        if reversal == 1 and fBp == 1:
                            iC = iC - 1  # set column to previous column
                            fB = fB + 1  # calculate number of filled Boxes

                ts[n, :] = [Box, iB, iC, TF, fB]

            elif TF == -1:

                # the Box index can not be zero
                if iB - 1 < 1:
                    iB = 1 + 1

                # check if there is a further 'O' in the trend
                if L[n] <= Boxes[iB - 1]:
                    [Box, iB, iC, TF, fB] = self._basic(L[n], iB, iC, TF, fB)

                else:

                    # check high for reversal
                    if H[n] >= Boxes[iB + reversal]:
                        TF = 1
                        [Box, iB, iC, TF, _] = self._basic(H[n], iB, iC, TF, fB)
                        iC = iC + 1  # go to next column
                        fB = iB - iBp  # calculate number of filled Boxes

                        # check for one-step-back
                        if reversal == 1 and fBp == 1:
                            iC = iC - 1  # set column to previous column
                            fB = fB + 1  # calculate number of filled Boxes

            ts[n, :] = [Box, iB, iC, TF, fB]

        return ts

    def _lohi(self, iD, Box, iB, iC, TF, fB):
        """
        logic for point and figure charts adapting the low/high method
        """
        H = self.ts['high']
        L = self.ts['low']

        Boxes = self.boxscale
        reversal = self.reversal

        ts = np.zeros([np.size(H), 5])

        # make the first entry right before the first change
        # otherwise filled boxes can be not correctly determined
        # in next iteration.
        ts[0: iD, :] = [Box, iB, iC, TF, fB]

        for n in range(iD, np.size(H)):

            iBp = iB  # Box index from previous iteration
            fBp = fB  # number of filled Boxes from previous iteration

            if TF == 1:

                # the Box index can not be zero
                if iB - reversal < 1:
                    iB = 1 + reversal

                # check for reversal
                if L[n] <= Boxes[iB - reversal]:
                    TF = -1
                    [Box, iB, iC, TF, _] = self._basic(L[n], iB, iC, TF, fB)
                    iC = iC + 1  # go to next column
                    fB = iBp - iB  # calculate number of filled Boxes

                    # check for one-step-back
                    if reversal == 1 and fBp == 1:
                        iC = iC - 1  # set column to previous column
                        fB = fB + 1  # calculate number of filled Boxes
                else:

                    # check if there is a further 'X' in the trend
                    if H[n] >= Boxes[iB + 1]:
                        [Box, iB, iC, TF, fB] = self._basic(H[n], iB, iC, TF, fB)

            elif TF == -1:

                # check for reversal
                if H[n] >= Boxes[iB + reversal]:
                    TF = 1
                    [Box, iB, iC, TF, _] = self._basic(H[n], iB, iC, TF, fB)
                    iC = iC + 1  # go to next column
                    fB = iB - iBp  # calculate number of filled Boxes

                    # check for one-step-back
                    if reversal == 1 and fBp == 1:
                        iC = iC - 1  # set column to previous column
                        fB = fB + 1  # calculate number of filled Boxes

                else:

                    # check if there is a further 'O' in the trend
                    if L[n] <= Boxes[iB - 1]:
                        [Box, iB, iC, TF, fB] = self._basic(L[n], iB, iC, TF, fB)

                    # else:  # do nothing
                    #   pass

            ts[n, :] = [Box, iB, iC, TF, fB]

        return ts

    def _hlc(self, iD, Box, iB, iC, TF, fB):
        """
        logic for point and figure charts adapting the high/low/close method
        """

        H = self.ts['high']
        L = self.ts['low']
        C = self.ts['close']

        Boxes = self.boxscale
        reversal = self.reversal

        ts = np.zeros([np.size(H), 5])

        # make the first entry right before the first change
        # otherwise filled boxes can be not correctly determined
        # in next iteration.
        ts[0: iD, :] = [Box, iB, iC, TF, fB]

        for n in range(iD, np.size(H)):

            iBp = iB  # Box index from previous iteration
            fBp = fB  # number of filled Boxes from previous iteration

            # trend is up
            if TF == 1:

                # check if there is a further 'X' in the trend
                if C[n] >= Boxes[iB + 1]:
                    [Box, iB, iC, TF, fB] = self._basic(H[n], iB, iC, TF, fB)

                else:

                    # the Box index can not be zero
                    if iB - reversal < 1:
                        iB = 1 + reversal

                    # check for reversal
                    if C[n] <= Boxes[iB - reversal]:
                        TF = -1
                        [Box, iB, iC, TF, _] = self._basic(L[n], iB, iC, TF, fB)
                        iC = iC + 1  # go to next column
                        fB = iBp - iB  # calculate number of filled Boxes

                        if reversal == 1 and fBp == 1:  # check for one-step-back
                            iC = iC - 1  # set column to previous column
                            fB = fB + 1  # calculate number of filled Boxes

                ts[n, :] = [Box, iB, iC, TF, fB]

            # trend is down
            elif TF == -1:

                # the Box index can not be zero
                if iB - 1 < 1:
                    iB = 1 + 1

                # check if there is a further 'O' in the trend
                if C[n] <= Boxes[iB - 1]:
                    [Box, iB, iC, TF, fB] = self._basic(L[n], iB, iC, TF, fB)

                else:

                    # check close for reversal
                    if C[n] >= Boxes[iB + reversal]:
                        TF = 1
                        [Box, iB, iC, TF, _] = self._basic(H[n], iB, iC, TF, fB)
                        iC = iC + 1  # go to next column
                        fB = iB - iBp  # calculate number of filled Boxes

                        # check for one-step-back
                        if reversal == 1 and fBp == 1:
                            iC = iC - 1  # set column to previous column
                            fB = fB + 1  # calculate number of filled Boxes

                ts[n, :] = [Box, iB, iC, TF, fB]

        return ts

    def _ohlc(self):
        """
        logic for point and figure charts adapting the open/high/low/close method
        """

        O = self.ts['open']
        H = self.ts['high']
        L = self.ts['low']
        C = self.ts['close']

        P = np.zeros(4 * np.size(C))

        tP = []
        counter = 0
        for n in range(counter, np.size(C)):

            if C[n] > O[n]:
                tP = [O[n], L[n], H[n], C[n]]

            elif C[n] < O[n]:
                tP = [O[n], H[n], L[n], C[n]]

            elif C[n] == O[n] and C[n] == L[n]:
                tP = [O[n], H[n], L[n], C[n]]

            elif C[n] == O[n] and C[n] == H[n]:
                tP = [O[n], L[n], H[n], C[n]]

            elif C[n] == O[n] and (H[n] + L[n]) / 2 > C[n]:
                tP = [O[n], H[n], L[n], C[n]]

            elif C[n] == O[n] and (H[n] + L[n]) / 2 < C[n]:
                tP = [O[n], L[n], H[n], C[n]]

            elif C[n] == O[n] and (H[n] + L[n]) / 2 == C[n]:

                if n > 1:
                    # if trend is uptrend
                    if C[n - 1] < C[n]:
                        tP = [O[n], H[n], L[n], C[n]]

                    # downtrend
                    elif C[n - 1] > C[n]:
                        tP = [O[n], L[n], H[n], C[n]]

                else:
                    tP = [O[n], H[n], L[n], C[n]]

            P[counter:counter + 4] = tP

            counter += 4

        # store initial close values temporary
        close = self.ts['close'].copy()

        # set the new time-series as close
        self.ts['close'] = P

        # determine the fist box entry
        [iD, Box, iB, iC, TF, fB] = self._get_first_trend()

        # restore initial close
        self.ts['close'] = close

        ts = np.zeros([np.size(P), 5])

        ts[0: iD, :] = [Box, iB, iC, TF, fB]

        for n in range(iD, len(P)):
            [Box, iB, iC, TF, fB] = self._basic(P[n], iB, iC, TF, fB)
            ts[n, :] = [Box, iB, iC, TF, fB]

        return ts

    def _get_pnf_timeseries(self):
        """
        builds time-series for point and figure chart
        """

        ts = self.ts

        date = ts['date']
        pfdate = date.copy()

        [iD, Box, iB, iC, TF, fB] = self._get_first_trend()

        if self.method == 'cl':
            ts = self._close(iD, Box, iB, iC, TF, fB)

        elif self.method == 'h/l':
            ts = self._hilo(iD, Box, iB, iC, TF, fB)

        elif self.method == 'l/h':
            ts = self._lohi(iD, Box, iB, iC, TF, fB)

        elif self.method == 'hlc':
            ts = self._hlc(iD, Box, iB, iC, TF, fB)

        elif self.method == 'ohlc':
            ts = self._ohlc()

            # reset the index and calculate missing datetimes
            if isinstance(self.ts['date'][0], np.datetime64):

                # extend initial index by 4 times and convert to seconds
                pfdate = np.repeat(pfdate, 4).astype('datetime64[s]')

                # find minimum in timedelta and assign to timestep
                timestep = np.min(np.diff(date))
                timestep = np.timedelta64(timestep, 's')

                # re-index the data
                counter = 0

                for n in range(0, np.size(date)):
                    pfdate[counter:counter + 4] = np.array([date[n],
                                                            date[n] + timestep * 0.25,
                                                            date[n] + timestep * 0.5,
                                                            date[n] + timestep * 0.75], dtype='datetime64[s]')
                    counter = counter + 4

            # date is not in datetime format, set index to integers
            else:
                pfdate = np.arange(0, np.shape(ts)[0])

        iTc = np.diff(np.append(0, ts[:, 3])).astype(bool)  # index of Trend change
        iBc = np.diff(np.append(0, ts[:, 1])).astype(bool)  # index of Box changes

        ic = np.logical_or(iBc, iTc)  # index of steps with changes

        ts[~ic, :] = np.nan  # set elements without action to NaN

        # index values cant be integer because of the nans in the arrays.
        pftseries = {'date': pfdate,
                     'box value': ts[:, 0],
                     'box index': ts[:, 1],
                     'column index': ts[:, 2],
                     'trend': ts[:, 3],
                     'filled boxes': ts[:, 4]}

        return pftseries

    def _get_column_entry_dates(self):

        date = self.pnf_timeseries['date']
        column_index = self.pnf_timeseries['column index']

        if self.time_step is not None:
            n = 0
            column_date_labels = []

            for d, c in zip(date, column_index):
                if c == n:
                    n = n + 1
                    d = np.datetime_as_string(d, unit=self.time_step)
                    d = d.replace('T', ' ')
                    column_date_labels.append(d)
        else:
            column_date_labels = None

        return column_date_labels

    def _pnf_timeseries2matrix(self):
        """
        builds Point and Figure matrix from Point and Figure time-series.
        """

        ts = self.pnf_timeseries
        boxes = self.boxscale

        iB = ts['box index'].copy()
        iC = ts['column index'].copy()
        TF = ts['trend'].copy()

        iNaN = np.isnan(iB)  # find indices of nan entries

        # remain entries without NaNs qne convert to int
        iB = iB[~iNaN].astype(int)
        iC = iC[~iNaN].astype(int)
        TF = TF[~iNaN].astype(int)

        mtx = np.zeros([np.size(boxes), iC[-1] + 1], dtype=int)

        # mark first box
        if TF[0] == 1:
            mtx[iB[0], 0] = 1
        elif TF[0] == -1:
            mtx[iB[1], 0] = -1

        # mark the other boxes
        for n in range(1, np.size(iB)):

            # positive trend goes on
            if TF[n - 1] == 1 and TF[n] == 1:
                mtx[iB[n - 1]:iB[n] + 1, iC[n]] = TF[n]

            # positive trend reverses
            elif TF[n - 1] == 1 and TF[n] == -1:
                mtx[iB[n]:iB[n - 1], iC[n]] = TF[n]

            # negative trend goes on
            elif TF[n - 1] == -1 and TF[n] == -1:
                mtx[iB[n]:iB[n - 1] + 1, iC[n]] = TF[n]

            # negative trend reverses
            elif TF[n - 1] == -1 and TF[n] == 1:
                mtx[iB[n - 1] + 1:iB[n] + 1, iC[n]] = TF[n]

        return mtx

    def get_breakouts(self):
        """
        Gets the breakouts of an PointFigureChart object

        Returns:
        ========

        breakouts: dict
            The dict contains following keys:
        breakouts['trend']:
            Array of int: 1 for bullish breakouts and -1 for bearish breakouts
        breakouts['type']:
            Array of str: continuation; fulcrum, resistance or reversal
        breakouts['hits']:
            Array of int: Values represent number of how often the
            line has been hit before the breakout.
        breakouts['width']:
            elements contain int of how long the line is
            between the first hit and the breakout.
        breakouts['outer width']:
            elements contain int of how long the line is from the breakout to
            the last filled box in previous columns on the same level.
            If there is no filled column the signal is counted as conti signal
            and the first column of the PointfigChart is used to calculate the
            outer width.
        """

        mtx = self.matrix

        a = np.zeros([np.size(mtx, 0), 1])
        b = mtx[:, 1:] - mtx[:, :-1]

        # find potential bullish breakouts
        T = np.concatenate((a, b), axis=1)
        T[(T < 1) | (mtx < 1)] = 0

        # row and col index of potential breakouts
        row_bull, col_bull = np.where(T == 1)

        # find potential bearish breakouts
        T = np.concatenate((a, b), axis=1)
        T[(T < -1) | (mtx > -1)] = 0

        # row and col index of potential breakouts
        row_bear, col_bear = np.where(T == -1)

        # initiate dictionary
        keys = ['trend', 'type', 'column index', 'box index', 'hits', 'width', 'outer width']
        bo = {}
        for key in keys:
            bo[key] = np.zeros(np.size(row_bull) + np.size(row_bear)).astype(int)
        # bo['trend'] = bo['trend'].astype(str)
        bo['type'] = bo['type'].astype(str)

        # assign trends
        bo['trend'][0:np.size(row_bull)] = 1
        bo['trend'][np.size(row_bull):np.size(row_bull) + np.size(row_bear)] = -1

        # bullish breakouts
        if np.any(row_bull):

            for n in range(0, np.size(row_bull)):

                bo['box index'][n] = row_bull[n]
                bo['column index'][n] = col_bull[n]

                hRL = mtx[row_bull[n] - 1, 0:col_bull[n] + 1]  # horizontal resistance line
                boL = mtx[row_bull[n], 0:col_bull[n] + 1]  # breakout line

                if np.any(np.where(hRL == -1)):
                    i = np.where(hRL == -1)[0][-1]
                else:
                    i = -1

                if np.any(np.where(hRL == 1)):
                    k = np.where(hRL == 1)[0]
                else:
                    k = 0

                if np.any(np.where(k > i)):
                    k = k[np.where(k > i)]

                # find type of signal
                z = 0
                if np.any(np.where(boL[:-1] != 0)) and np.size(k) >= 2:
                    z = np.where(boL[:-1] != 0)[0][-1]
                    bo['outer width'][n] = k[-1] - z + 1

                elif np.size(k) >= 2:
                    bo['outer width'][n] = k[-1] + 1

                if z >= 1:

                    if mtx[row_bull[n], z - 1] == 0 and mtx[row_bull[n], z] == 1:
                        bo['type'][n] = 'resistance'

                    elif mtx[row_bull[n], z - 1] == 1 and mtx[row_bull[n], z] == 1:
                        bo['type'][n] = 'resistance'

                    elif mtx[row_bull[n], z - 1] == -1 and mtx[row_bull[n], z] == -1:
                        bo['type'][n] = 'fulcrum'

                    elif mtx[row_bull[n], z - 1] == -1 and mtx[row_bull[n], z] == 1:
                        bo['type'][n] = 'reversal'

                    elif mtx[row_bull[n], z - 1] == 0 and mtx[row_bull[n], z] == -1:
                        bo['type'][n] = 'reversal'

                    elif mtx[row_bull[n], z - 1] == 1 and mtx[row_bull[n], z] == -1:
                        bo['type'][n] = 'reversal'

                    elif mtx[row_bull[n], z - 1] == 0 and mtx[row_bull[n], z] == 0:
                        bo['type'][n] = 'conti'

                elif z == 0:

                    if mtx[row_bull[n], z] == 0:
                        bo['type'][n] = 'conti'

                    elif mtx[row_bull[n], z] == 1:
                        bo['type'][n] = 'conti'

                    elif mtx[row_bull[n], z] == -1:
                        bo['type'][n] = 'reversal'

                if np.size(k) >= 2:
                    bo['hits'][n] = np.size(k)
                    bo['width'][n] = k[-1] - k[0] + 1

                # find smaller breakouts within other breakouts
                if np.size(k) > 2:

                    for p in range(1, np.size(k) - 1):
                        bo['trend'] = np.append(bo['trend'], 1)
                        bo['type'] = np.append(bo['type'], bo['type'][n])
                        bo['column index'] = np.append(bo['column index'], bo['column index'][n])
                        bo['box index'] = np.append(bo['box index'], bo['box index'][n])
                        bo['hits'] = np.append(bo['hits'], np.sum(mtx[row_bull[n] - 1, k[p]:k[-1] + 1]))
                        bo['width'] = np.append(bo['width'], [k[-1] - k[p] + 1])
                        bo['outer width'] = np.append(bo['outer width'], bo['outer width'][n])

        # bearish breakouts
        if np.any(row_bear):

            for n in range(0, np.size(row_bear)):

                bo['box index'][np.size(row_bull) + n] = row_bear[n]
                bo['column index'][np.size(row_bull) + n] = col_bear[n]

                hRL = mtx[row_bear[n] + 1, 0:col_bear[n] + 1]  # horizontal resistance line
                boL = mtx[row_bear[n], 0:col_bear[n] + 1]  # breakout line

                if np.any(np.where(hRL == 1)):
                    i = np.where(hRL == 1)[0][-1]

                else:
                    i = -1

                if np.any(np.where(hRL == -1)):
                    k = np.where(hRL == -1)[0]

                else:
                    k = 0

                if np.any(np.where(k > i)):
                    k = k[np.where(k > i)]

                # find type of signal
                z = 0
                if np.any(np.where(boL[:-1] != 0)) and np.size(k) >= 2:
                    z = np.where(boL[:-1] != 0)[0][-1]
                    bo['outer width'][np.size(row_bull) + n] = k[-1] - z + 1

                elif np.size(k) >= 2:
                    bo['outer width'][np.size(row_bull) + n] = k[-1] + 1

                if z >= 1:

                    if mtx[row_bear[n], z - 1] == 0 and mtx[row_bear[n], z] == -1:
                        bo['type'][np.size(row_bull) + n] = 'resistance'

                    elif mtx[row_bear[n], z - 1] == -1 and mtx[row_bear[n], z] == -1:
                        bo['type'][np.size(row_bull) + n] = 'resistance'

                    elif mtx[row_bear[n], z - 1] == 1 and mtx[row_bear[n], z] == 1:
                        bo['type'][np.size(row_bull) + n] = 'reversal'

                    elif mtx[row_bear[n], z - 1] == 1 and mtx[row_bear[n], z] == -1:
                        bo['type'][np.size(row_bull) + n] = 'reversal'

                    elif mtx[row_bear[n], z - 1] == 0 and mtx[row_bear[n], z] == 1:
                        bo['type'][np.size(row_bull) + n] = 'reversal'

                    elif mtx[row_bear[n], z - 1] == -1 and mtx[row_bear[n], z] == 1:
                        bo['type'][np.size(row_bull) + n] = 'reversal'

                    elif mtx[row_bear[n], z - 1] == 0 and mtx[row_bear[n], z] == 0:
                        bo['type'][np.size(row_bull) + n] = 'conti'

                elif z == 0:

                    if mtx[row_bear[n], z] == 0:
                        bo['type'][np.size(row_bull) + n] = 'conti'
                    elif mtx[row_bear[n], z] == -1:
                        bo['type'][np.size(row_bull) + n] = 'conti'
                    elif mtx[row_bear[n], z] == 1:
                        bo['type'][np.size(row_bull) + n] = 'reversal'

                if np.size(k) >= 2:
                    bo['hits'][np.size(row_bull) + n] = np.size(k)
                    bo['width'][np.size(row_bull) + n] = k[-1] - k[0] + 1

                # find smaller breakouts within other breakouts
                if np.size(k) > 2:

                    for p in range(1, np.size(k) - 1):
                        bo['trend'] = np.append(bo['trend'], -1)
                        bo['type'] = np.append(bo['type'], bo['type'][np.size(row_bull) + n])
                        bo['column index'] = np.append(bo['column index'], bo['column index'][np.size(row_bull) + n])
                        bo['box index'] = np.append(bo['box index'], bo['box index'][np.size(row_bull) + n])
                        bo['hits'] = np.append(bo['hits'], np.abs(np.sum(mtx[row_bear[n] + 1, k[p]:k[-1] + 1])))
                        bo['width'] = np.append(bo['width'], [k[-1] - k[p] + 1])
                        bo['outer width'] = np.append(bo['outer width'], bo['outer width'][np.size(row_bull) + n])

        # find index without entries:
        x = np.argwhere(bo['hits'] == 0)
        for key in bo.keys():
            bo[key] = np.delete(bo[key], x)

        # sort order: col , row, hits
        T = np.column_stack((bo['column index'], bo['box index'], bo['hits']))
        idx = np.lexsort((T[:, 2], T[:, 1], T[:, 0]))
        for key, value in bo.items():
            bo[key] = bo[key][idx]

        self.breakouts = bo

        return bo

    def get_trendlines(self, length=4, mode='strong'):
        """
        Gets trendlines of an PointfigChart object

        Parameter:
        ==========

        length: int
            minimum length for trendlines default(4).
        mode: str
            'strong' or 'weak' default('strong')
            Strong trendlines break is the line hits a filled box whereas weak lines
            break after a breakout in the other direction occurred above a bearish
            resistance line or below a bullish support line.

        Returns:
        ========

        trendlines: dict
            trendlines['bounded']:
                Array of str: Trendlines are bounded 'internal' or 'external'.
            trendlines['type']: str
                Array of str: Trendlines are 'bullish support' or 'bearish resistance' lines.
            trendlines['length']: int
                Array of int: Length of the trendline.
            trendlines['column index']: int
                Array of int: Index of column where the trendline starts.
            trendlines['box index']: int
                Array of int: Index of row where the trendline starts.
        """

        mtx = self.matrix.copy()

        # correct/initiate minimum length for trendlines:
        if mode == 'weak' and length <= 3:
            length = 4
            warn('Set trendline length to 4. Minimum Length for trendlines of mode=weak is 4.')

        elif mode == 'strong' and length <= 2:
            length = 3
            warn('Set trendline length to 3. Minimum Length for trendlines of mode=strong is 3.')

        # if there is just 1 box filled in first column of mtx add another one
        # to prevent letting trendlines run out of range.
        if np.sum(np.abs(mtx[:, 0])) == 1:

            if np.sum(mtx[:, 0]) > 0:
                idx = np.where(mtx[:, 0] != 0)[0][-1]
                mtx[idx - 1, 0] = 1

            elif np.sum(mtx[:, 0]) > 0:
                idx = np.where(mtx[:, 0] != 0)[0][0]
                mtx[idx + 1, 0] = 1

        # find high and low index for each column; sign indicates trend direction
        T = [np.repeat([np.arange(1, np.size(mtx, 0) + 1, 1)], np.size(mtx, 1), axis=0)][0].transpose() * mtx
        T = np.abs(T)

        ceil = np.zeros(np.size(T, 1)).astype(int)
        floor = np.zeros(np.size(T, 1)).astype(int)

        for n in range(0, np.size(T, 1)):

            high = np.max(T[:, n])
            low = np.min(T[np.where(T[:, n] != 0), n])

            ceil[n] = np.where(T[:, n] == high)[0][0]

            if np.sign(mtx[ceil[n], n]) < 0:
                ceil[n] = ceil[n] * (-1)

            floor[n] = np.where(T[:, n] == low)[0][0]

            if np.sign(mtx[floor[n], n]) < 0:
                floor[n] = floor[n] * (-1)

        # extent mtx in variable T to prevent that trendlines run out of the
        # matrix the offset will be later removed from the data
        offset = np.size(mtx, 1)

        T = np.vstack((np.zeros([np.size(mtx, 1), np.size(mtx, 1)]),
                       mtx,
                       np.zeros([np.size(mtx, 1), np.size(mtx, 1)])
                       )).astype(int)

        T = np.hstack((T, np.zeros([np.size(T, 0), length - 1])))

        # add ones in the last column to stop the latest trendlines
        T = np.hstack((T, np.ones([np.size(T, 0), 1])))

        # new indices after extension
        ceil[ceil > 0] = ceil[ceil > 0] + offset
        ceil[ceil < 0] = ceil[ceil < 0] - offset

        floor[floor > 0] = floor[floor > 0] + offset
        floor[floor < 0] = floor[floor < 0] - offset

        # initiate tl_mtx as matrix containing all possible trendlines
        tl_mtx = np.zeros([np.size(T, 0), np.size(T, 1)])

        if mode == 'weak':

            # initiate matrix for breakpoints for trendlines
            brkpt = np.zeros([np.size(T, 0), np.size(T, 1)])
            # brkpt[:,-1] = 1

            # check if breakouts have been initiated earlier
            if self.breakouts is None:
                bo = self.get_breakouts()

            else:
                bo = self.breakouts

            col = bo['column index'][bo['trend'] == 1]
            row = bo['box index'][bo['trend'] == 1] + offset
            brkpt[row, col] = 1

            col = bo['column index'][bo['trend'] == -1]
            row = bo['box index'][bo['trend'] == -1] + offset
            brkpt[row, col] = -1

            # fill tl_mtx with the length of the trendline at the position of
            # the starting point

            # bearish resistance line starts above every X-column and moves downwards
            # with an 45°-angle until a buy signal is hit or above the line
            for n in range(0, np.size(floor)):

                if ceil[n] > 0:
                    k = ceil[n] + 1
                    col = n

                    while np.sum(brkpt[k:-1, col]) <= 0 and col < np.size(brkpt, 1) - 1:
                        col = col + 1
                        k = k - 1

                    tl_mtx[np.abs(ceil[n]) + 1, n] = n - col

            # bullish support line starts below every O-column and moves upwards with
            # an 45°-angle until a sell signal is hit or below the line
            for n in range(0, np.size(ceil)):

                if floor[n] < 0:
                    k = np.abs(floor[n]) - 1
                    col = n

                    while np.sum(brkpt[0:k, col]) >= 0 and col < np.size(brkpt, 1) - 1:
                        col = col + 1
                        k = k + 1

                    tl_mtx[np.abs(floor[n]) - 1, n] = col - n

            tl_mtx = tl_mtx.astype(int)

            # set all trendlines to zero which are shorter than the minimum length
            tl_mtx[np.abs(tl_mtx) < length] = 0

        # find strong trendlines that will be broken once hit a filled box
        elif mode == 'strong':

            # bearish resistance line starts above every X-column and moves downwards
            # with an 45°-angle until there is any entry different from zero in trendline_mtx
            for n in range(0, np.size(floor)):

                if ceil[n] > 0:
                    k = ceil[n] + 1
                    col = n

                    while T[k, col] == 0:
                        col = col + 1
                        k = k - 1

                    tl_mtx[np.abs(ceil[n]) + 1, n] = n - col

            # bullish support line starts below every O-column and moves upwards with
            # an 45°-angle until there is any entry different from zero in trendline_mtx
            for n in range(0, np.size(ceil)):

                if floor[n] < 0:
                    k = np.abs(floor[n]) - 1
                    col = n

                    while T[k, col] == 0:
                        col = col + 1
                        k = k + 1

                    tl_mtx[np.abs(floor[n]) - 1, n] = col - n

            tl_mtx = tl_mtx.astype(int)
            tl_mtx[np.abs(tl_mtx) < length] = 0

        # counter for the loop to exit if an unexpected case occurred
        loop_run = 0

        # find first trendline
        col = 0
        while np.sum(np.abs(tl_mtx[:, col])) == 0:
            col = col + 1

        # initiate variables for the lookup of external trendlines
        iB = np.argwhere(tl_mtx[:, col] != 0)[0]  # index of last Box
        tF = np.sign(tl_mtx[iB, col])[0]  # TrendFlag
        span = np.abs(tl_mtx[iB, col])[0]  # length of trendline

        tl_vec = np.zeros(np.size(tl_mtx, 1))  # tl_vec: 1d vector of trendlines
        tl_vec[col] = span * tF

        while col + span <= np.size(T, 1) - length - 1 and loop_run <= np.size(T, 1):

            # v_down contains trendlines in the current interval moving downwards
            # v_up contains trendlines in the current interval moving upwards
            v_down = tl_mtx[:, col:col + span].copy()
            v_down[v_down > 0] = 0
            v_down = np.sum(v_down, 0)
            v_up = tl_mtx[:, col:col + span].copy()
            v_up[v_up < 0] = 0
            v_up = np.sum(v_up, 0)

            # remove possible trendlines which are touching occupied boxes within
            # the current interval (necessary for "weak" mode - no impact on strong
            # mode)
            if tF == 1:

                for x in range(0, np.size(v_down)):

                    if v_down[x] != 0:
                        a = np.size(v_down) - np.where(v_down == v_down[x])[0][0]
                        b = np.where(v_down == v_down[x])[0][0]
                        z = np.flipud(np.eye(a))
                        iB = np.argwhere(tl_mtx[:, col + b] != 0)[0][0]
                        check = T[iB - np.size(z, 0) + 1:iB + 1, col + b: col + b + np.size(z, 0)]

                        if np.any(check * z):
                            v_down[x] = 0

            elif tF == -1:

                for x in range(0, np.size(v_up)):

                    if v_up[x] != 0:

                        a = np.size(v_up) - np.where(v_up == v_up[x])[0][0]
                        b = np.where(v_up == v_up[x])[0][0]
                        z = np.eye(a)
                        iB = np.argwhere(tl_mtx[:, col + b] != 0)[0][0]  # index of last Box
                        check = T[iB - 1:iB + np.size(z, 0) - 1, col + b: col + b + np.size(z, 0)]

                        if np.any(check * z):
                            v_up[x] = 0

            if tF == 1:

                # direction of current trendline is up
                # create array containing the position(index+1) of elements of v_down
                # which are not zero. The length of the corresponding line is added to
                # the position. If the number is greater than length of variable, the
                # trendline does leave the interval
                check = (v_down < 0) * np.arange(1, np.size(v_down) + 1, 1) + np.abs(v_down)

                if np.any(v_down) == 1:  # there is a reversal trendline in the interval

                    # check if the reversal trendline leaves the interval
                    if np.any(check > np.size(v_down)) == 1:
                        col = col + np.where(check == np.max(check))[0][0]
                        span = np.sum(np.abs(tl_mtx[:, col]))
                        tF = np.sign(np.sum(tl_mtx[:, col]))
                        tl_vec[col] = span * tF

                    # the reversal trendline does not leave the interval
                    else:
                        tl_mtx[:, col + 1:col + span - 1] = 0

                # there is no reversal trendline in the interval
                elif np.any(check) == 0:

                    # go to next trendline regardless of their direction
                    col = col + np.size(check)
                    span = 1

                    while np.sum(np.sum(np.abs(tl_mtx[:, col:col + span]), 0)) == 0:
                        span = span + 1

                    col = col + span - 1
                    span = np.abs(np.sum(tl_mtx[:, col]))
                    tF = np.sign(np.sum(tl_mtx[:, col]))
                    tl_vec[col] = span * tF

            elif tF == -1:

                # direction of current trendline is down
                # create array containing the position(index+1) of elements of v_down
                # which are not zero. The length of the corresponding line is added to
                # the position. If the number is greater than length of variable, the
                # trendline does leave the interval
                check = (v_up > 0) * np.arange(1, np.size(v_up) + 1, 1) + v_up

                # there is a reversal trendline in the interval
                if np.any(v_up) == 1:

                    # check if the reversal trendline leaves the interval
                    if np.any(check > np.size(v_up)) == 1:
                        col = col + np.where(check == np.max(check))[0][0]
                        span = np.sum(np.abs(tl_mtx[:, col]))
                        tF = np.sign(np.sum(tl_mtx[:, col]))
                        tl_vec[col] = span * tF

                    # the reversal trendline does not leave the interval
                    else:
                        tl_mtx[:, col + 1:col + span - 1] = 0

                # there is no reversal trendline in the interval
                elif np.any(check) == 0:

                    # go to next trendline despite of their direction
                    col = col + np.size(check)
                    span = 1

                    while np.sum(np.sum(np.abs(tl_mtx[:, col:col + span]), 0)) == 0:
                        span = span + 1

                    col = col + span - 1
                    span = np.abs(np.sum(tl_mtx[:, col]))
                    tF = np.sign(np.sum(tl_mtx[:, col]))
                    tl_vec[col] = span * tF

            loop_run += 1

            if loop_run >= np.size(T, 1):
                # raise IndexError('An unexpected case occurred during evaluating the trendlines.')
                break

        # prepare returned variable for trendlines
        row, col = np.where(tl_mtx != 0)

        tlines = {'bounded': np.zeros(np.size(col)).astype(str),
                  'type': np.zeros(np.size(col)).astype(str),
                  'length': np.zeros(np.size(col)).astype(int),
                  'column index': np.zeros(np.size(col)).astype(int),
                  'box index': np.zeros(np.size(col)).astype(int)
                  }

        for n in range(0, np.size(col)):

            # check for bounding
            if tl_vec[col[n]] != 0:
                tlines['bounded'][n] = 'external'
            else:
                tlines['bounded'][n] = 'internal'

            tlines['column index'][n] = col[n]
            tlines['box index'][n] = row[n] - offset

            # the latest trendlines can be shorter than the minimum length.
            # correct the latest trendlines to the actual length.
            if np.abs(tl_mtx[row[n], col[n]]) + col[n] >= np.size(mtx, 1):
                tlines['length'][n] = np.abs(tl_mtx[row[n], col[n]]) - length + 1

            else:
                tlines['length'][n] = np.abs(tl_mtx[row[n], col[n]])

            if tl_mtx[row[n], col[n]] > 0:
                tlines['type'][n] = 'bullish support'

            else:
                tlines['type'][n] = 'bearish resistance'

        # find  and delete index without entries
        x = np.argwhere(tlines['length'] == 0)
        for key in tlines.keys():
            tlines[key] = np.delete(tlines[key], x)

        # sort columns
        idx = np.argsort(tlines['column index'])
        for key, value in tlines.items():
            tlines[key] = tlines[key][idx]

        self.trendlines = tlines

        return tlines

    def _get_midpoints(self):
        """
        Calculates the midpoints for every column of an Point and Figure Chart
        """

        boxes = self.boxscale
        mtx = self.matrix

        points = np.zeros(np.size(mtx, 1))

        for n in range(0, np.size(mtx, 1)):

            column = mtx[:, n]
            column = np.where(column != 0)[0]
            column = boxes[column]

            if self.method == 'log':

                i = np.floor(np.size(column) / 2).astype(int) - 1

                if i < (np.size(column) / 2) - 1:
                    center_value = column[i + 1]
                else:
                    center_value = np.exp((np.log(column[i]) + np.log(column[i + 1])) / 2)

            else:
                i = np.floor(np.size(column) / 2).astype(int) - 1

                if i < (np.size(column) / 2) - 1:
                    center_value = column[i + 1]
                else:
                    center_value = column[i] + (column[i + 1] - column[i]) / 2

            points[n] = center_value

        self.column_midpoints = points

        return points

    def midpoints(self):
        if self.column_midpoints is None:
            self._get_midpoints()

        self.indicator['Midpoints'] = self.column_midpoints

        return self.column_midpoints

    def sma(self, period):
        """
         Calculates the simple moving average for every column of an Point and Figure Chart
         """
        label = f'SMA({period})'

        if self.column_midpoints is None:
            self.column_midpoints = self._get_midpoints()
            values = self.column_midpoints
        else:
            values = self.column_midpoints

        ma = np.zeros(len(values))
        ma[:] = np.nan

        if len(ma) >= period:

            for n in range(period - 1, len(values)):
                ma[n] = np.mean(values[n - period + 1:n + 1])

        self.indicator[label] = ma

        return ma

    def ema(self, period):
        """
        Calculates the exponential moving average for every column of an Point and Figure Chart
        """
        label = f'EMA({period})'

        if self.column_midpoints is None:
            self.column_midpoints = self._get_midpoints()
            values = self.column_midpoints
        else:
            values = self.column_midpoints

        ma = np.zeros(len(values))
        ma[:] = np.nan

        if len(ma) >= period:

            ma[period - 1] = np.sum(values[0:period]) / period
            k = 2 / (period + 1)

            for n in range(period, len(values)):
                ma[n] = k * (values[n] - ma[n - 1]) + ma[n - 1]

        self.indicator[label] = ma

        return ma

    def bollinger(self, period, factor):
        """
        Calculates the bollinger bands for every column of an Point and Figure Chart
        """

        label = f'Bollinger({period},{factor})'

        mtx = self.matrix

        upper_band = np.zeros(np.size(mtx, 1))
        upper_band[:] = np.nan

        bb_l = np.zeros(np.size(mtx, 1))
        bb_l[:] = np.nan

        std = np.zeros(np.size(mtx, 1))
        std[:] = np.nan

        if f'SMA({period})' in self.indicator:
            ma = self.indicator[f'SMA({period})']
        else:
            ma = self.sma(period)
            self.indicator.pop(f'SMA({period})')

        mp = self.column_midpoints

        if len(upper_band) >= period:

            for n in range(period - 1, len(std)):
                std[n] = np.std(mp[n - period + 1:n + 1])

        upper_band = ma + factor * std
        lower_band = ma - factor * std

        self.indicator[label + '-upper'] = upper_band
        self.indicator[label + '-lower'] = lower_band

        return upper_band, lower_band

    def donchian(self, period, ignore_columns=0):
        """
        Calculates the Donchian channels for every column of an Point and Figure Chart.
        ignore_column is the number of columns that will be ignored at the end
        and it's equivalent to shifting the channels to the right.
        """
        label = f'Donchian({period},{ignore_columns})'

        matrix = np.abs(self.matrix).astype('float')
        boxscale = self.boxscale

        boxscale = boxscale.reshape(len(boxscale), 1)

        boxscale = np.repeat(boxscale, repeats=np.shape(matrix)[1], axis=1)

        matrix = np.multiply(boxscale, matrix)

        matrix[matrix == 0] = np.nan

        high = np.nanmax(matrix, 0)
        low = np.nanmin(matrix, 0)

        donchian_channel_middle = np.zeros(len(high))
        donchian_channel_middle[:] = np.nan

        donchian_channel_upper = np.zeros(len(high))
        donchian_channel_upper[:] = np.nan

        donchian_channel_lower = np.zeros(len(low))
        donchian_channel_lower[:] = np.nan

        if len(donchian_channel_upper) >= period:

            for n in range(period - 1, len(donchian_channel_upper)):
                donchian_channel_upper[n] = np.max(high[n - period + 1:n + 1])
                donchian_channel_lower[n] = np.min(low[n - period + 1:n + 1])
                # donchian_channel_middle[n] = (donchian_channel_upper[n]-donchian_channel_lower[n])/2

        if ignore_columns > 0 and ignore_columns <= len(donchian_channel_upper):
            donchian_channel_upper = np.append(np.repeat(np.nan, ignore_columns),
                                               donchian_channel_upper[:-ignore_columns])
            donchian_channel_lower = np.append(np.repeat(np.nan, ignore_columns),
                                               donchian_channel_lower[:-ignore_columns])
            # donchian_channel_middle = np.append(np.repeat(np.nan, ignore_columns), donchian_channel_middle[:-ignore_columns])

        self.indicator[label + '-upper'] = donchian_channel_upper
        self.indicator[label + '-lower'] = donchian_channel_lower

        return donchian_channel_upper, donchian_channel_lower

    def psar(self, step, leap):
        """
        Calculates the parabolic Stop and Reverse (pSAR) for every column of an Point and Figure Chart
        """

        label = f'pSAR({step},{leap})'
        boxes = self.boxscale
        mtx = self.matrix

        # check length her and leave function
        if np.size(mtx, 1) <= 2:
            psar = np.zeros(np.size(mtx, 1))
            psar[:] = np.nan
            self.indicator[label] = psar

            return psar

        mtx = [np.repeat([boxes], np.size(mtx, 1), axis=0)][0].transpose() * mtx
        mtx = np.abs(mtx)

        high = np.zeros(np.size(mtx, 1))
        low = np.zeros(np.size(mtx, 1))

        for n in range(0, np.size(mtx, 1)):
            t = mtx[:, n]
            high[n] = np.max(t)
            t[t == 0] = np.max(t)
            low[n] = np.min(mtx[:, n])

        psar = np.zeros(np.size(high))
        ep = np.zeros(np.size(high))
        diff = np.zeros(np.size(high))
        prod = np.zeros(np.size(high))
        trendflag = np.zeros(np.size(high))
        accFactor = np.zeros(np.size(high))
        trendlength = np.zeros(np.size(high))
        trendlength[0] = 1

        if high[0] > high[2]:

            psar[0] = high[0]
            ep[0] = low[0]
            trendflag[0] = -1

        else:
            psar[0] = low[0]
            ep[0] = high[0]
            trendflag[0] = 1

        diff[0] = ep[0] - psar[0]
        accFactor[0] = step
        prod[0] = diff[0] * accFactor[0]

        for n in range(1, np.size(high)):

            if trendflag[n - 1] == 1 and prod[n - 1] + psar[n - 1] > low[n]:
                psar[n] = ep[n - 1]
            elif trendflag[n - 1] == -1 and prod[n - 1] + psar[n - 1] < high[n]:
                psar[n] = ep[n - 1]
            else:
                psar[n] = psar[n - 1] + prod[n - 1]

            if psar[n] < high[n]:
                trendflag[n] = 1
            elif psar[n] > low[n]:
                trendflag[n] = -1

            if trendflag[n] == 1 and high[n] > ep[n - 1]:
                ep[n] = high[n]
            elif trendflag[n] == 1 and high[n] <= ep[n - 1]:
                ep[n] = ep[n - 1]
            elif trendflag[n] == -1 and low[n] < ep[n - 1]:
                ep[n] = low[n]
            elif trendflag[n] == -1 and low[n] >= ep[n - 1]:
                ep[n] = ep[n - 1]

            if trendflag[n] == trendflag[n - 1]:
                trendlength[n] = trendlength[n - 1] + 1
                if accFactor[n - 1] == leap:
                    accFactor[n] = leap
                elif trendflag[n] == 1 and ep[n] > ep[n - 1]:
                    accFactor[n] = accFactor[n - 1] + step
                elif trendflag[n] == 1 and ep[n] <= ep[n - 1]:
                    accFactor[n] = accFactor[n - 1]
                elif trendflag[n] == -1 and ep[n] < ep[n - 1]:
                    accFactor[n] = accFactor[n - 1] + step
                elif trendflag[n] == -1 and ep[n] >= ep[n - 1]:
                    accFactor[n] = accFactor[n - 1]
            else:
                accFactor[n] = step
                trendlength[n] = 1

            diff[n] = ep[n] - psar[n]
            prod[n] = accFactor[n] * diff[n]

        psar = psar * trendflag

        self.indicator[label] = psar

        return psar

    def next_simple_signal(self):

        next_buy = np.nan
        next_sell = np.nan

        # last trend need to be identified from pnfts
        idx = np.where(~np.isnan(self.pnf_timeseries['trend']))[0][-1]

        last_trend = int(self.pnf_timeseries['trend'][idx])

        if np.shape(self.matrix)[1] >= 3:

            mtx = self.matrix.copy()
            mtx = mtx[:, -3:]

            x_col_1 = np.where(mtx[:, 0] == 1)[0]
            x_col_2 = np.where(mtx[:, 1] == 1)[0]
            x_col_3 = np.where(mtx[:, 2] == 1)[0]

            o_col_1 = np.where(mtx[:, 0] == -1)[0]
            o_col_2 = np.where(mtx[:, 1] == -1)[0]
            o_col_3 = np.where(mtx[:, 2] == -1)[0]

            if last_trend == 1:

                if np.any(x_col_2):
                    idx = x_col_2[-1]
                else:
                    idx = x_col_1[-1]

                if idx + 1 > x_col_3[-1]:
                    # if idx  > x_col_3[-1]:
                    next_buy = self.boxscale[idx + 1]
                else:
                    next_buy = np.nan

                if np.any(o_col_3):
                    idx = o_col_3[0]
                else:
                    idx = o_col_2[0]

                next_sell = self.boxscale[idx - 1]

            elif last_trend == -1:

                if np.any(o_col_2):
                    idx = o_col_2[0]
                else:
                    idx = o_col_1[0]

                if idx - 1 < o_col_3[0]:
                    # if idx < o_col_3[0]:
                    next_sell = self.boxscale[idx - 1]
                else:
                    next_sell = np.nan

                if np.any(x_col_3):
                    idx = x_col_3[-1]
                else:
                    idx = x_col_2[-1]

                next_buy = self.boxscale[idx + 1]

        # print('Next Buy: ', next_buy)
        # print('Next Sell: ', next_sell)

        return next_buy, next_sell

    def multiple_top_buy(self, label, multiple):

        max_width = 2 * multiple - 1

        array = np.zeros(len(self.pnf_timeseries['box index']))
        array[:] = np.nan

        x = ((self.breakouts['trend'] == 1)
             & (self.breakouts['width'] <= max_width)
             & (self.breakouts['hits'] == multiple))

        col = self.breakouts['column index'][x]
        row = self.breakouts['box index'][x]

        for r, c in zip(row, col):
            col_idx = (self.pnf_timeseries['column index'] == c)
            row_idx = self.pnf_timeseries['box index'][col_idx]
            ts_idx = int(row_idx[row_idx >= r][0])
            x = ((self.pnf_timeseries['box index'] == ts_idx) & (self.pnf_timeseries['column index'] == c))
            array[x] = self.boxscale[r]

            self.buys[label] = array

    def multiple_bottom_sell(self, label, multiple):

        max_width = 2 * multiple - 1

        array = np.zeros(len(self.pnf_timeseries['box index']))
        array[:] = np.nan

        x = ((self.breakouts['trend'] == -1)
             & (self.breakouts['width'] <= max_width)
             & (self.breakouts['hits'] == multiple))

        col = self.breakouts['column index'][x]
        row = self.breakouts['box index'][x]

        for r, c in zip(row, col):
            col_idx = (self.pnf_timeseries['column index'] == c)
            row_idx = self.pnf_timeseries['box index'][col_idx]
            ts_idx = int(row_idx[row_idx <= r][0])
            x = ((self.pnf_timeseries['box index'] == ts_idx) & (self.pnf_timeseries['column index'] == c))
            array[x] = self.boxscale[r]

            self.sells[label] = array

    def double_top_buy(self):

        self.multiple_top_buy(label='DTB', multiple=2)

    def double_bottom_sell(self):

        self.multiple_bottom_sell(label='DBS', multiple=2)

    def triple_top_buy(self):

        self.multiple_top_buy(label='TTB', multiple=3)

    def triple_bottom_sell(self):

        self.multiple_bottom_sell(label='TBS', multiple=3)

    def _coordinates2plot_grid(self, array):
        """
        Converts price coordinates to the plot grid.
        """

        coords_on_grid = np.full(len(array), np.nan)
        boxscale = self.boxscale
        scaling = self.scaling

        if scaling == 'log':
            base = 1 + pnf.boxsize / 100

        for num, val in enumerate(array):

            if not np.isnan(val):
                if any(np.argwhere(boxscale <= val)):
                    index = np.argwhere(boxscale <= val)[-1]
                else:
                    index = 0

                point_1 = boxscale[index]
                point_2 = boxscale[index + 1]
                point_3 = val

                if scaling == 'log':
                    dist = np.log(point_3 / point_1) / np.log(base)
                else:
                    dist = (point_3 - point_1) / (point_2 - point_1)

                coords_on_grid[num] = np.round(index + dist, 3)[0]

        return coords_on_grid

    def _change_color_opacity(self, index):
        """
        Change the opacity of a color from a Matplotlib color scale defined in self.indicator_colors.
        """

        color = list(self.indicator_colors(index))
        color[3] = self.indicator_fillcolor_opacity
        color = tuple(color)

        return color

    def _indicator_plotting_preparations(self):
        """
        Converts the indicator coordinates to the plotting grid and cuts off nan from
        the indicator arrays and adjust the matrix length if cut2indicator is True.
        """

        plot_indicator = self.indicator.copy()

        # find latest starting indicator
        indicator_cut_length = 0
        if self.cut2indicator is True:

            non_nan_pos = []
            for key in plot_indicator:
                array = plot_indicator[key]
                index = np.argwhere(~np.isnan(array))[0]
                non_nan_pos.append(index)

            indicator_cut_length = np.max(non_nan_pos)

        # convert indicator coordinates to the plotting grid
        for key in plot_indicator:

            plot_indicator[key] = plot_indicator[key][indicator_cut_length:]

            if 'pSAR' not in key:
                plot_indicator[key] = self._coordinates2plot_grid(plot_indicator[key])

            if 'pSAR' in key:
                sign = np.sign(plot_indicator[key])
                plot_indicator[key] = self._coordinates2plot_grid(np.abs(plot_indicator[key])) * sign

        self.plot_indicator = plot_indicator
        self.cut2indicator_length = indicator_cut_length

    def _set_margins(self):
        """
        Sets the margins for th eplot figure based on the length of the x- and y-ticks
        """

        # separate method for margin determination
        if self.margin_bottom is None:
            if self.column_axis is True:
                if self.time_step == 'D':
                    self.margin_bottom = 0.85
                elif self.time_step == 'm':
                    self.margin_bottom = 1.42
                elif self.time_step is None:
                    self.margin_bottom = 0.1
            else:
                self.margin_bottom = 0.1

        # calculate side margins based on number of max y-label characters

        # if margins_left or margin_right is True

        if self.left_axis is True or self.right_axis is True:
            y_ticks = np.arange(0, np.shape(self.plot_matrix)[0], 1)
            y_ticklabels = self.plot_boxscale[y_ticks].astype('str')
            max_y_tick_length = np.max(list(map(len, y_ticklabels)))

        if self.margin_left is None:
            if self.left_axis is True:
                self.margin_left = (max_y_tick_length + 0.5) / 10
            else:
                self.margin_left = 0.1

        if self.margin_right is None:
            if self.right_axis is True:
                self.margin_right = (max_y_tick_length + 0.5) / 10
            else:
                self.margin_right = 0.1

    def _evaluate_figure_size_and_set_plot_options(self):
        """
        Calculates the figure size and sets the parameters for the plot based on the size of the matrix
        """

        # figure height based on matrix dim-0 length and margins
        figure_height_array = (np.array(self.plotsize_options['box_height']) * np.shape(self.plot_matrix)[0]
                               + self.margin_bottom + self.margin_top)

        # figure size based on matrix dim-1 length and margins
        figure_width_array = (np.array(self.plotsize_options['box_height']) * (np.shape(self.plot_matrix)[1]
                                                                               + self.add_empty_columns) + self.margin_left + self.margin_right)

        # figure size based on matrix_min_width
        figure_width_matrix_min_width_array = (np.multiply(np.array(self.plotsize_options['box_height']),
                                                           np.array(self.plotsize_options['matrix_min_width'])
                                                           + self.add_empty_columns)
                                               + self.margin_left + self.margin_right)

        index = np.array(self.plotsize_options['matrix_min_width']) > np.shape(self.plot_matrix)[1]

        figure_width_array[index] = figure_width_matrix_min_width_array[index]

        n = 0
        if self.size == 'auto':

            while figure_width_array[n] >= self.max_figure_width or figure_height_array[n] >= self.max_figure_height:
                n = n + 1
                if n == len(figure_width_array):
                    n = n - 1
                    break

        elif self.size == 'huge':
            n = 0

        elif self.size == 'large':
            n = 1

        elif self.size == 'medium':
            n = 2

        elif self.size == 'small':
            n = 3

        elif self.size == 'tiny':
            n = 4

        if self.size != 'auto':
            self.size = self.plotsize_options['size'][n]

        self.figure_height = figure_height_array[n]
        self.figure_width = figure_width_array[n]

        if self.box_height is None:
            self.box_height = self.plotsize_options['box_height'][n]

        if self.marker_linewidth is None:
            self.marker_linewidth = self.plotsize_options['marker_linewidth'][n]

        if self.grid_linewidth is None:
            self.grid_linewidth = self.plotsize_options['grid_linewidth'][n]

        if self.x_label_step is None:
            self.x_label_step = self.plotsize_options['x_label_step'][n]

        if self.y_label_step is None:
            self.y_label_step = self.plotsize_options['y_label_step'][n]

        self.matrix_min_width = self.plotsize_options['matrix_min_width'][n]

        if self.grid is None:
            self.grid = self.plotsize_options['grid'][n]

    def _evaluate_optimal_legend_position(self):

        legend_matrix = np.hstack((self.plot_matrix,
                                   np.zeros([np.shape(self.plot_matrix)[0], self.add_empty_columns])))

        h1 = np.floor(np.shape(legend_matrix)[0] / 2).astype('int')
        w1 = np.floor(np.shape(legend_matrix)[1] / 2).astype('int')

        mod_h = np.mod(np.shape(legend_matrix)[0] / 2, 1)
        mod_w = np.mod(np.shape(legend_matrix)[1] / 2, 1)

        if mod_h == 0 and mod_w == 0:
            h2 = h1 + 1
            w2 = w1 + 1

        elif mod_h != 0 and mod_w != 0:
            h2 = h1
            w2 = w1

        elif mod_h == 0 and mod_w != 0:
            h2 = h1 + 1
            w2 = w1

        elif mod_h != 0 and mod_w == 0:
            h2 = h1
            w2 = w1 + 1

        bot_left = np.abs(legend_matrix)[0:h1, 0:w1]
        bot_right = np.abs(legend_matrix)[0:h1, w2:]
        top_left = np.abs(legend_matrix)[h2:, 0:w1]
        top_right = np.abs(legend_matrix)[h1:, w2:]

        matrix_quadrant_sums = [np.sum(top_left), np.sum(top_right), np.sum(bot_left), np.sum(bot_right)]

        quadrant = np.argmin(matrix_quadrant_sums)  # returns the first occurrence of a min value

        legend_positions = ['upper left', 'upper right', 'lower left', 'lower right']

        self.legend_position = legend_positions[quadrant]

    def _prepare_variables_for_plotting(self):
        """
        Prepares matrix and indicator for plotting. Stores the cut_off_indices in attributes.
        The cut_off indices are needed to plot signals and trendlines
        """

        self._indicator_plotting_preparations()

        if self.cut2indicator is True:
            self.plot_matrix = self.matrix[:, self.cut2indicator_length:]
        else:
            self.plot_matrix = self.matrix

        if np.nonzero(np.sum(np.abs(self.plot_matrix), 1))[0][0] - 3 <= 0:
            self.matrix_bottom_cut_index = 0
        else:
            self.matrix_bottom_cut_index = np.nonzero(np.sum(np.abs(self.matrix), 1))[0][0] - 3

        self.matrix_top_cut_index = np.nonzero(np.sum(np.abs(self.matrix), 1))[0][-1] + 4

        self.plot_matrix = self.plot_matrix[self.matrix_bottom_cut_index: self.matrix_top_cut_index, :]
        self.plot_boxscale = self.boxscale[self.matrix_bottom_cut_index: self.matrix_top_cut_index]

        self._set_margins()
        self._evaluate_figure_size_and_set_plot_options()

        if np.shape(self.plot_matrix)[1] < self.matrix_min_width:
            extension_length = self.matrix_min_width - np.shape(self.plot_matrix)[1]
        else:
            extension_length = 0

        # extend the matrix with zeros if dim-1 is too short
        self.plot_matrix = np.hstack((self.plot_matrix, np.zeros([np.shape(self.plot_matrix)[0], extension_length])))

        # extend indicator with np.nan by extension_length
        extension = np.full([1, extension_length], np.nan)[0]

        for key in self.plot_indicator:

            if 'pSAR' not in key:
                self.plot_indicator[key] = np.hstack(
                    (self.plot_indicator[key], extension)) - self.matrix_bottom_cut_index

            if 'pSAR' in key:
                sign = np.sign(self.plot_indicator[key])
                sign = np.hstack((sign, extension))
                self.plot_indicator[key] = (np.abs(
                    np.hstack((self.plot_indicator[key], extension))) - self.matrix_bottom_cut_index) * sign

        # calculate ticks and ticklabels

        # prepare y-ticks
        self.plot_y_ticks = np.arange(0, np.shape(self.plot_matrix)[0], self.y_label_step)
        self.plot_y_ticklabels = self.plot_boxscale[self.plot_y_ticks]

        # prepare x-ticks
        self.plot_column_label = self.column_labels[::-self.x_label_step]

        x_ticks = np.arange(np.size(self.column_labels))
        self.plot_column_index = x_ticks[::-self.x_label_step] + 0.5

        if self.legend_position is None:
            self._evaluate_optimal_legend_position()

    def _create_figure_and_axis(self):
        """
        Creates the figure and axis objects.
        """

        # plt.ioff()  # necessary to supress output in jupyter notebooks

        # calculate axis positioning
        left = self.margin_left / self.figure_width
        right = self.margin_right / self.figure_width
        bottom = self.margin_bottom / self.figure_height
        top = self.margin_top / self.figure_height
        width = 1 - left - right
        height = 1 - bottom - top

        # initiate figure
        self.fig = plt.figure(self.title, figsize=(self.figure_width, self.figure_height))

        # first axis creates the frame for the chart.
        self.ax1 = self.fig.add_axes((0, 0, 1, 1))
        self.ax1.axis('off')
        self.ax1.set_yticks([])
        self.ax1.set_xticks([])
        self.ax1.get_tightbbox()

        # second axis is where the plotting takes place
        self.ax2 = self.fig.add_axes((left, bottom, width, height))

        if self.left_axis is True:
            self.ax2.set_yticks(self.plot_y_ticks)
            self.ax2.set_yticklabels(self.plot_y_ticklabels, fontsize=self.label_fontsize)
        else:
            self.ax2.set_yticks([])
            self.ax2.set_yticklabels([])

        self.ax2.set_ylim(bottom=-0.5, top=np.shape(self.plot_matrix)[0] - 0.5)

        # third axis is to allow y-ticks with labels on the ight of the chart
        self.ax3 = self.ax2.twinx()
        self.ax3.set_xticks([])

        if self.right_axis is True:
            self.ax3.set_yticks(self.plot_y_ticks)
            self.ax3.set_yticklabels(self.plot_y_ticklabels, fontsize=self.label_fontsize)
        else:
            self.ax3.set_yticks([])
            self.ax3.set_yticklabels([])

        self.ax3.set_ylim(bottom=-0.5, top=np.shape(self.plot_matrix)[0] - 0.5)

        if self.column_axis is True:
            self.ax2.set_xticks(self.plot_column_index)
            self.ax2.set_xticklabels(self.plot_column_label, rotation=90, ha='center', fontsize=self.label_fontsize)
        else:
            self.ax2.set_xticks([])
            self.ax2.set_xticklabels([])

        self.ax2.set_xlim(left=0, right=np.shape(self.plot_matrix)[1] + self.add_empty_columns)

    def _plot_grid(self):
        """
        Plots a grid to the PointFigureChart figure
        """

        for n in np.arange(np.shape(self.plot_matrix)[0]):
            x1 = 0
            x2 = np.shape(self.plot_matrix)[1] + self.add_empty_columns
            self.ax2.plot((x1, x2), (n + 0.5, n + 0.5), color=self.grid_color, lw=self.grid_linewidth)

        for n in np.arange(np.shape(self.plot_matrix)[1] + self.add_empty_columns):
            y1 = 0 - 0.5
            y2 = np.shape(self.plot_matrix)[0] - 0.5
            self.ax2.plot((n, n), (y1, y2), color=self.grid_color, lw=self.grid_linewidth)

    def _plot_markers(self):
        """
        Plots Point and Figure symbols (X and O) to the PointFigureChart figure
        """

        x_box, x_col = np.where(self.plot_matrix > 0)
        o_box, o_col = np.where(self.plot_matrix < 0)

        x_col = x_col + 0.5
        o_col = o_col + 0.5

        space = 0.4  # spacer between symbols

        if self.show_markers is True:
            for n in range(0, np.size(x_col)):
                self.ax2.plot((x_col[n] - space, x_col[n] + space), (x_box[n] - space, x_box[n] + space),
                              color=self.x_marker_color,
                              lw=self.marker_linewidth)
                self.ax2.plot((x_col[n] + space, x_col[n] - space), (x_box[n] - space, x_box[n] + space),
                              color=self.x_marker_color,
                              lw=self.marker_linewidth)

            for n in range(0, np.size(o_col)):
                circle = plt.Circle((o_col[n], o_box[n]), space, color=self.o_marker_color, lw=self.marker_linewidth,
                                    fill=False)
                self.ax2.add_artist(circle)

    def _plot_trendlines(self):
        """
        plots 45 degree trendlines to the PointFigureChart figure
        """
        if self.show_trendlines == 'external':
            trendline_modus = 'external'
        elif self.show_trendlines == 'internal':
            trendline_modus = 'internal'
        else:
            trendline_modus = 'external'

        trendlines = self.trendlines

        for n in range(0, np.size(trendlines['column index'])):

            if trendlines['bounded'][n] == trendline_modus:

                if trendlines['type'][n] == 'bullish support':
                    c = trendlines['column index'][n]
                    r = trendlines['box index'][n] - self.matrix_bottom_cut_index
                    r_floor = r + 0.5
                    r_ceill = r - 0.5
                    self.ax2.plot((c, c + 1), (r_ceill, r_floor), color='b', lw=self.marker_linewidth)
                    k = 1

                    while k < trendlines['length'][n]:
                        c = c + 1
                        r_floor = r_ceill
                        r_ceill = r_ceill + 1
                        k = k + 1
                        self.ax2.plot((c, c + 1), (r_floor + 1, r_ceill + 1), color='b', lw=self.marker_linewidth)

                elif trendlines['type'][n] == 'bearish resistance':

                    c = trendlines['column index'][n]
                    r = trendlines['box index'][n] - self.matrix_bottom_cut_index
                    r_floor = r + 0.5
                    r_ceill = r - 0.5

                    self.ax2.plot((c, c + 1), (r_floor, r_ceill,), color='r', lw=self.marker_linewidth)
                    k = 1

                    while k < trendlines['length'][n]:
                        c = c + 1
                        r_ceill = r_floor
                        r_floor = r_floor - 1
                        k = k + 1

                        self.ax2.plot((c, c + 1), (r_ceill - 1, r_floor - 1), color='r', lw=self.marker_linewidth)

    def _plot_breakouts(self):
        """
        Plots breakout lines to the PointFigureChart figure
        """

        if self.breakouts is None:
            self.breakouts = self.get_breakouts()
            bo = self.breakouts
        else:
            bo = self.breakouts

        for i, row, col, width in zip(np.arange(0, np.size(bo['column index'])),
                                      bo['box index'],
                                      bo['column index'],
                                      bo['width']):
            if bo['trend'][i] == 1:
                y = row - 0.5 - self.matrix_bottom_cut_index
                x1 = col + 1
                x2 = x1 - width
                self.ax2.plot((x1, x2), (y, y), color=self.bullish_breakout_color, lw=self.marker_linewidth)

            elif bo['trend'][i] == -1:
                y = row + 0.5 - self.matrix_bottom_cut_index
                x1 = col + 1
                x2 = x1 - width
                self.ax2.plot((x1, x2), (y, y), color=self.bearish_breakout_color, lw=self.marker_linewidth)

    def _plot_indicator(self):
        """
        Plots applied indicator to the PointFigureChart figure
        """

        # calculate x coordinates for indicator
        x_coordinates = np.arange(np.shape(self.plot_matrix)[1]) + 0.5

        indicator_keys = []

        for key in self.plot_indicator.keys():

            if 'Bollinger' in key:
                if 'upper' in key:
                    indicator_keys.append(key.split('-')[0])

            if 'Donchian' in key:
                if 'upper' in key:
                    indicator_keys.append(key.split('-')[0])

            elif 'pSAR' in key:
                indicator_keys.append(key)

            elif not 'Bollinger' in key and not 'Donchian' in key and not 'pSAR' in key:
                indicator_keys.append(key)

        color_index = 0
        legend_entries = []

        # plot indicator
        for indicator in indicator_keys:

            if 'Bollinger' in indicator or 'Donchian' in indicator:
                bbu = self.plot_indicator[indicator + '-upper']
                bbl = self.plot_indicator[indicator + '-lower']
                self.ax2.plot(x_coordinates, bbu, '-', color=self.indicator_colors(color_index),
                              linewidth=self.marker_linewidth)
                self.ax2.plot(x_coordinates, bbl, '-', color=self.indicator_colors(color_index),
                              linewidth=self.marker_linewidth,
                              label=indicator)
                self.ax2.fill_between(x_coordinates, bbu, bbl,
                                      color=self._change_color_opacity(color_index))  # , alpha=1)

                fillcolor = self._change_color_opacity(color_index)
                legend_symbol_bollinger = Line2D([], [], color=fillcolor,
                                                 marker='s',
                                                 linestyle='None',
                                                 markeredgewidth=1,
                                                 markersize=8,
                                                 label=indicator,
                                                 fillstyle='full',
                                                 markeredgecolor=self.indicator_colors(color_index))
                legend_entries.append(legend_symbol_bollinger)
                color_index += 1

            if 'pSAR' in indicator:
                sign = np.sign(self.plot_indicator[indicator])
                psar = np.abs(self.plot_indicator[indicator])

                for val, c, tf in zip(psar, x_coordinates, sign):

                    if tf == 1:
                        self.ax2.scatter(c, val, s=self.marker_linewidth * 5, marker='o',
                                         color=self.indicator_colors(color_index))
                    elif tf == -1:
                        self.ax2.scatter(c, val, s=self.marker_linewidth * 5, marker='o',
                                         color=self.indicator_colors(color_index + 1))

                legend_symbol_psar = Line2D([], [], color=self.indicator_colors(color_index),
                                            marker='o',
                                            linestyle='None',
                                            markeredgewidth=0.1,
                                            markersize=5,
                                            label=indicator,
                                            fillstyle='left',
                                            markerfacecoloralt=self.indicator_colors(color_index + 1))
                legend_entries.append(legend_symbol_psar)
                color_index += 2

            if not 'Bollinger' in indicator and not 'Donchian' in indicator and not 'pSAR' in indicator:
                self.ax2.plot(x_coordinates, self.plot_indicator[indicator], '-',
                              color=self.indicator_colors(color_index),
                              linewidth=self.marker_linewidth)
                legend_symbol = Line2D([], [], color=self.indicator_colors(color_index),
                                       linestyle='-', label=indicator)
                legend_entries.append(legend_symbol)
                color_index += 1

            self.legend_entries = legend_entries

    def _assemble_plot_chart(self):
        self._prepare_variables_for_plotting()
        self._create_figure_and_axis()

        # plot grid
        if self.grid is True:
            self._plot_grid()

        # plot points and figures
        if self.show_markers is True:
            self._plot_markers()

        # plot breakouts
        if self.show_breakouts is True:
            self._plot_breakouts()

        # plot trendlines
        # check if  trendlines are there
        if self.show_trendlines == 'external' or self.show_trendlines == 'internal':
            self._plot_trendlines()
        elif self.show_trendlines == 'both':
            self.show_trendlines = 'external'
            self._plot_trendlines()
            self.show_trendlines = 'internal'
            self._plot_trendlines()
            self.show_trendlines = 'both'

        # plot indicator
        self._plot_indicator()

        # plot volume at price
        if self.vap != {}:
            self._plot_volume_at_price()

        if self.legend_entries is not None:
            self.ax2.legend(handles=self.legend_entries, fontsize=self.legend_fontsize, loc=self.legend_position)

        plt.title(self.title, loc='left', fontsize=self.title_fontsize)

    def save(self, fname=None, dpi=None):

        if self.fig is None:
            self._assemble_plot_chart()

        if fname is None:
            fname = 'chart.png'

        if dpi is None:
            if self.size == 'tiny' or self.size == 'small':
                dpi = 1200
            else:
                dpi = 600

        self.fig.savefig(fname=fname, dpi=dpi, bbox_inches='tight', pad_inches=0)

    def show(self):

        if self.fig is None:
            self._assemble_plot_chart()

        plt.show()

    def __str__(self):

        mtx = self.matrix
        boxes = self.boxscale.copy()

        print_mtx = self.matrix.copy()
        last_trendline = []
        last_trendline_length = []

        if self.trendlines is not None:
            tlines = self.trendlines

            for n in range(0, np.size(tlines['column index'])):

                if tlines['bounded'][n] == 'external':

                    if tlines['type'][n] == 'bullish support':

                        last_trendline = 'bullish support'
                        last_trendline_length = tlines['length'][n]
                        c = tlines['column index'][n]
                        r = tlines['box index'][n]

                        if mtx[r, c] == 0:
                            print_mtx[r, c] = 2
                        k = 1

                        while k < tlines['length'][n] and c < np.shape(mtx)[1] - 1:

                            c = c + 1
                            r = r + 1
                            k = k + 1

                            if mtx[r, c] == 0:
                                print_mtx[r, c] = 2

                    elif tlines['type'][n] == 'bearish resistance':

                        last_trendline = 'bearish resistance'
                        last_trendline_length = tlines['length'][n]
                        c = tlines['column index'][n]
                        r = tlines['box index'][n]

                        if mtx[r, c] == 0:
                            print_mtx[r, c] = -2
                        k = 1

                        while k < tlines['length'][n] and c < np.shape(mtx)[1] - 1:

                            c = c + 1
                            r = r - 1
                            k = k + 1

                            if mtx[r, c] == 0:
                                print_mtx[r, c] = -2

        columns = 30
        total_columns = np.shape(mtx)[1]

        if columns >= total_columns:
            columns = total_columns

        print_mtx = print_mtx[:, -columns:]
        idx = np.where(np.sum(np.abs(mtx[:, -columns:]), axis=1) != 0)[0]
        boxes = boxes[idx]
        print_mtx = print_mtx[idx, :]

        print_mtx = np.flipud(print_mtx).astype(str)
        boxes = np.flipud(boxes).astype(str)

        n = 0
        table = []
        for m in range(len(boxes)):

            row = print_mtx[m, :]
            row = [s.replace('0', '.') for s in row]
            row = [s.replace('-1', 'O') for s in row]
            row = [s.replace('1', 'X') for s in row]
            row = [s.replace('-2', '*') for s in row]
            row = [s.replace('2', '*') for s in row]
            row = np.hstack((boxes[m], row, boxes[m]))

            if n == 0:
                table = row
            else:
                table = np.vstack((table, row))
            n += 1

        table = tabulate(table, tablefmt='simple')

        print(self.title)
        print(table)

        if self.trendlines is not None:
            print(f'last trendline: {last_trendline} line of length {last_trendline_length}')
        return f'printed {columns}/{total_columns} columns.'


if __name__ == '__main__':
    from testdata import dataset

    data = dataset('^SPX')

    pnf = PointFigureChart(ts=data, method='h/l', reversal=2, boxsize=2, scaling='log', title='^SPX')
    pnf.get_trendlines(length=4, mode='weak')
    pnf.show_trendlines = 'external'
    pnf.show()

