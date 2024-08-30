from dotenv import load_dotenv

load_dotenv()

import os
import polars as pl


def test_env_threads():

    # The only lib where we can actually check that this
    # works is polars. All the others we have to set and pray
    # This *should persist across spanws

    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["VECLIB_MAXIMUM_THREADS"] == "1"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "1"
    assert pl.thread_pool_size() == 1
