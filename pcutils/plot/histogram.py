import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional
from matplotlib.pyplot import Axes
from matplotlib.text import Text
from matplotlib.colors import LogNorm, Colormap
from matplotlib.backend_bases import LocationEvent
from matplotlib.collections import QuadMesh
import matplotlib.pyplot as plt
from matplotlib import colormaps
from math import floor

cmap = colormaps['viridis']
cmap.set_under('white')

#Utility functions
def clamp_low(x: float, edge: float) -> float:
    return x if x > edge else edge

def clamp_hi(x: float, edge: float) -> float:
    return x if x < edge else edge

def clamp_range(xrange: tuple[float, float], min_max: tuple[float, float]):
    return (clamp_low(xrange[0], min_max[0]), clamp_hi(xrange[1], min_max[1]))

'''
Hist1D, Hist2D
Dataclasses storing histogram data (name, counts per bin, bin edges)
When going to plot use the following idioms:

Hist1D:
    matplotlib.pyplot.stairs(hist.counts, hist.bins)

Hist2D:
    matplotlib.pyplot.pcolormesh(hist.x_bins, hist.y_bins, hist.counts)
'''
@dataclass
class Hist1D:
    '''
    Container wrapping a numpy array used to store histogram data and retrieve histogram statistics
    '''
    name: str
    counts: NDArray[np.float64]
    bins: NDArray[np.float64]
    bin_width: float

    def get_bin(self, x: float) -> Optional[int]:
        if x < self.bins.min() or x > self.bins.max():
            return None
        
        return int(floor((x - self.bins[0]) / self.bin_width))

    #returns (integral, mean, std_dev)
    def stats_for_range(self, xrange: tuple[float, float]) -> Optional[tuple[float, float, float]]:
        clamped_range = clamp_range(xrange, (self.bins.min(), self.bins.max()))
        bin_min = self.get_bin(clamped_range[0])
        bin_max = self.get_bin(clamped_range[1])
        if bin_min is None or bin_max is None:
            return None
        integral = np.sum(self.counts[bin_min:bin_max])
        mean = np.average(self.bins[bin_min:bin_max], weights=self.counts[bin_min:bin_max])
        variance = np.average((self.bins[bin_min:bin_max] - mean)**2.0, weights=self.counts[bin_min:bin_max])
        return (integral, mean, np.sqrt(variance))

@dataclass
class Hist2D:
    '''
    Container wrapping a numpy array used to store two-dimensional histogram data and retrieve histogram statistics
    '''
    name: str
    counts: NDArray[np.float64]
    x_bins: NDArray[np.float64]
    y_bins: NDArray[np.float64]
    x_bin_width: float
    y_bin_width: float

    def get_bin(self, coords: tuple[float, float]) -> tuple[int, int]:
        if coords[0] < self.x_bins.min() or coords[0] > self.x_bins.max():
            return None

        y_bin = int(floor((coords[1] - self.y_bins[0]) / self.y_bin_width))
        x_bin = int(floor((coords[0] - self.x_bins[0]) / self.x_bin_width))
        return (x_bin, y_bin)

    #returns (integral, mean x, std_dev x, mean y, std_dev y)
    def stats_for_range(self, ranges: tuple[tuple[float, float], tuple[float, float]]) -> Optional[tuple[float, float, float, float, float]]:
        clamped_x_range = clamp_range(ranges[0], (self.x_bins.min(), self.x_bins.max()))
        clamped_y_range = clamp_range(ranges[1], (self.y_bins.min(), self.y_bins.max()))
        bin_min = self.get_bin((clamped_x_range[0], clamped_y_range[0]))
        bin_max = self.get_bin((clamped_x_range[1], clamped_y_range[1]))

        x_bin_range = np.arange(start=bin_min[0], stop=bin_max[0], step=1)
        y_bin_range = np.arange(start=bin_min[1], stop=bin_max[1], step=1)
        bin_mesh = np.ix_(y_bin_range, x_bin_range)

        integral = np.sum(self.counts[bin_mesh])
        mean_x = np.average(self.x_bins[bin_min[0]:bin_max[0]], weights=np.sum(self.counts.T[bin_min[0]:bin_max[0]], 1))
        mean_y = np.average(self.y_bins[bin_min[1]:bin_max[1]], weights=np.sum(self.counts[bin_min[1]:bin_max[1]], 1))
        var_x = np.average((self.x_bins[bin_min[0]:bin_max[0]] - mean_x)**2.0, weights=np.sum(self.counts.T[bin_min[0]:bin_max[0]], 1))
        var_y = np.average((self.y_bins[bin_min[1]:bin_max[1]] - mean_y)**2.0, weights=np.sum(self.counts[bin_min[1]:bin_max[1]], 1))
        return (integral, mean_x, mean_y, np.sqrt(var_x), np.sqrt(var_y))




class Histogrammer:
    '''
    Histogrammer is a wrapper around a dictionary of str->Hist1D|Hist2D
    A new histogram can be added to the dictionary using the add_hist1d/add_hist2d methods. The name passed to
    these methods is used as the key for the dictionary. To add data to the histograms use the fill_hist1d/fill_hist2d methods.
    The fill methods accept arrays of data, and this is by intent. It would not be efficient to fill the histograms point by point. Rather, prefer
    passing entire data sets (like dataframe columns). Finally, to retrieve a histogram (for plotting, etc), use the get_hist1d/get_hist2d methods.
    Prefer the getters over direct access to the underlying dictionary as the getters perfom some error checking.

    ## Note
    Should be pickle-able -> We can save histograms in a concise binary way
    '''
    def __init__(self):
        self.histograms: dict[str, Hist1D | Hist2D] = {}
        self.axes: dict[Axes, tuple[str, Optional[Text]]] = {}
        self.figures: dict[str, bool] = {} #used to indicate if callbacks have been bound for that figure

    def add_hist1d(self, name: str, bins: int, range: tuple[float, float]):
        if name in self.histograms:
            print(f'Overwriting histogram named {name} in Histogrammer.add_histogram!')

        hist = Hist1D(name, np.empty(0), np.empty(0), np.abs(range[0] - range[1])/float(bins))
        hist.counts, hist.bins = np.histogram(a=[], bins=bins, range=range)
        self.histograms[name] = hist

    def add_hist2d(self, name: str, bins: tuple[int, int], ranges: tuple[tuple[float, float], tuple[float, float]]):
        if name in self.histograms:
            print(f'Overwriting histogram named {name} in Histogrammer.add_histogram!')

        hist = Hist2D(name, np.empty(0), np.empty(0), np.empty(0), np.abs(ranges[0][0] - ranges[0][1])/float(bins[0]), np.abs(ranges[1][0] - ranges[1][1])/float(bins[1]))
        hist.counts, hist.x_bins, hist.y_bins = np.histogram2d(x=[], y=[], bins=bins, range=ranges)
        hist.counts = hist.counts.T
        self.histograms[name] = hist

    def fill_hist1d(self, name: str, data: np.ndarray) -> bool:
        if name not in self.histograms:
            return False

        hist = self.histograms[name]
        if type(hist) is not Hist1D:
            return False

        hist.counts = hist.counts + np.histogram(a=data, bins=hist.bins)[0]
        return True

    def fill_hist2d(self, name: str, x_data: np.ndarray, y_data: np.ndarray) -> bool:
        if name not in self.histograms:
            return False
        
        hist = self.histograms[name]
        if type(hist) is not Hist2D:
            return False
        counts, x_edges, y_edges = np.histogram2d(x_data.flatten(), y_data.flatten(), bins=(hist.x_bins, hist.y_bins))
        hist.counts += counts.T
        return True

    def get_hist1d(self, name: str) -> Optional[Hist1D]:
        if name not in self.histograms:
            return None
        
        hist = self.histograms[name]
        if type(hist) is not Hist1D:
            return None
        else:
            return hist

    def get_hist2d(self, name: str) -> Optional[Hist2D]:
        if name not in self.histograms:
            return None
        
        hist = self.histograms[name]
        if type(hist) is not Hist2D:
            return None
        else:
            return hist

    def on_axes_enter_hist1d(self, event: LocationEvent):
        data = self.axes[event.inaxes]
        xrange = event.inaxes.get_xbound()
        yrange = event.inaxes.get_ybound()
        stats = self.histograms[data[0]].stats_for_range(xrange)
        if data[1] is not None:
            data[1].remove()

        draw_x = xrange[1] - 0.25 * np.abs(xrange[0] - xrange[1])
        draw_y = yrange[1] - 0.25 * np.abs(yrange[0] - yrange[1])
        self.axes[event.inaxes] = (data[0], event.inaxes.text(draw_x, draw_y, f'Integral: {stats[0]}\nCentroid: {stats[1]:.3f}\nSigma: {stats[2]:.3f}'))
        plt.draw()

    def on_axes_enter_hist2d(self, event: LocationEvent):
        data = self.axes[event.inaxes]
        xrange = event.inaxes.get_xbound()
        yrange = event.inaxes.get_ybound()
        stats = self.histograms[data[0]].stats_for_range((xrange, yrange))
        if data[1] is not None:
            data[1].remove()

        draw_x = xrange[1] - 0.25 * np.abs(xrange[0] - xrange[1])
        draw_y = yrange[1] - 0.25 * np.abs(yrange[0] - yrange[1])
        self.axes[event.inaxes] = (data[0], event.inaxes.text(draw_x, draw_y,
                        f'Integral: {stats[0]}\nCentroid X: {stats[1]:.3f}\nCentroid Y: {stats[2]:.3f}\nSigma X: {stats[3]:.3f}\nSigma Y: {stats[4]:.3f}',
                        color='black'))
        plt.draw()

    def on_axes_enter(self, event: LocationEvent):
        if event.inaxes not in self.axes:
            return
        
        if type(self.histograms[self.axes[event.inaxes][0]]) is Hist1D:
            self.on_axes_enter_hist1d(event)
        elif type(self.histograms[self.axes[event.inaxes][0]]) is Hist2D:
            self.on_axes_enter_hist2d(event)

    def on_axes_leave(self, event: LocationEvent):
        if event.inaxes not in self.axes:
            return
        data = self.axes[event.inaxes]
        if data[1] is None:
            return
        data[1].remove()
        self.axes[event.inaxes] = (data[0], None)
        plt.draw()

    def connect_mpl_callbacks(self, axis: Axes):
        if not hasattr(axis.figure, '_suptitle'):
            axis.figure.suptitle(f'Figure {len(self.figures)}')
        elif axis.figure._suptitle in self.figures:
            return

        self.figures[axis.figure._suptitle] = True
        axis.figure.canvas.mpl_connect('axes_enter_event', self.on_axes_enter)
        axis.figure.canvas.mpl_connect('axes_leave_event', self.on_axes_leave)

    def draw_hist1d(self, name: str, axis: Axes):
        if name not in self.histograms:
            return

        hist = self.histograms[name]
        if type(hist) is not Hist1D:
            return

        axis.stairs(hist.counts, hist.bins)
        self.axes[axis] = (name, None)
        self.connect_mpl_callbacks(axis)
            
        
    def draw_hist2d(self, name: str, axis: Axes, log_z: bool = False, cmap = 'viridis') -> QuadMesh:
        if name not in self.histograms:
            return
        
        hist = self.histograms[name]
        if type(hist) is not Hist2D:
            return
        mesh = None
        if log_z:
            mesh = axis.pcolormesh(hist.x_bins, hist.y_bins, hist.counts, cmap=cmap, norm=LogNorm())
        else:
            mesh = axis.pcolormesh(hist.x_bins, hist.y_bins, hist.counts, cmap=cmap, vmin=1.0e-6)
        self.axes[axis] = (name, None)
        self.connect_mpl_callbacks(axis)
        return mesh
