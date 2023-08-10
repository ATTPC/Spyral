import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from pcutils.core.get_trace import GetTrace
from pcutils.core.get_event import GetEvent, read_get_event
from pcutils.core.hardware_id import hardware_id_from_array
from pcutils.hdf.TPCH5_utils import load_trace, get_first_last_event_num

def GetTraceTest(h5_Path, event_num):
    metas, traces = load_trace(h5_Path, event_num = event_num)

    for i in range(100):

        trace_num = np.random.randint(low = 0, high = len(traces)-1)
        #trace_num = 183

        Trace = GetTrace(data = traces[trace_num], id = hardware_id_from_array(metas[trace_num]))
        Trace.find_peaks()

        if Trace.get_number_of_peaks() >= 1:
            break

    print(trace_num)
    #print(traces[trace_num])

    print(Trace.get_number_of_peaks())

    for Peak in Trace.peaks:
        print(Peak.positive_inflection, Peak.negative_inflection, Peak.uncorrected_amplitude)

    plt.figure(figsize = (8, 6))
    plt.plot(Trace.raw_data, label = 'Raw data')
    plt.plot(Trace.corrected_data, label = 'Data w/o baseline')
    #plt.plot(Trace.smoothed_output, label = 'Smoothed data w/o baseline')
    plt.scatter(np.array([Peak.centroid for Peak in Trace.get_peaks()]),
                np.array([Peak.amplitude for Peak in Trace.get_peaks()]),
                marker = '*', 
                color = 'green',
                zorder = 3)
    for Peak in Trace.get_peaks():
        plt.axvspan(xmin = Peak.positive_inflection,
                    xmax = Peak.negative_inflection,
                    ymin = min(Trace.corrected_data),
                    ymax = max(Trace.corrected_data),
                    alpha = 0.5)
    plt.grid()
    plt.legend()
    plt.title(trace_num)
    plt.show()

def GetEventTest(h5_Path, event_num):
    Event = read_get_event(h5_Path, event_num)
    print(Event.name)

def main():
    h5_Path = '/mnt/analysis/e20009/e20009_Turi/run_0348.h5'
    event_num = 146674
    GetTraceTest(h5_Path, event_num)
    GetEventTest(h5_Path, event_num)

if __name__ == '__main__':
    main()
    print('All done!')
