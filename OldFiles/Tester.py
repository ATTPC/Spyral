import timeit

s = """\
import numpy as np
import pandas as pd
import time
import sys
sys.path.insert(0, '../TPC-utils')
from tpc_utils import search_high_res
from TPCH5_utils import get_first_last_event_num, load_trace

PATH = '../run_0231.h5'

event_ind = 1060

meta, all_traces = load_trace(PATH, event_ind)
all_traces = all_traces.astype(np.float64)
"""

cy = timeit.timeit('CT_utils_CY.flag_ct(all_traces, meta)',
                    setup = 'import CT_utils_CY; '+s,
                    number = 100)
py = timeit.timeit('CT_utils_PY.flag_ct(all_traces, meta)',
                    setup = 'import CT_utils_PY; '+s,
                    number = 100)

print(cy, py)
print('Cython is {}x faster'.format(py/cy))
