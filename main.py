from spyral.core.config import load_config
# from spyral.run import run_spyral
from spyral.run_parallel import run_spyral_parallel
import sys
from pathlib import Path
import time
import datetime

def main(config_path: str):
    config = load_config(Path(config_path))
    run_spyral_parallel(config)

if __name__ == "__main__":
    start = time.time()
    if len(sys.argv) < 2:
        print('spyral requires a configuration file!')
    else:
        main(sys.argv[1])
    print(f'Time elapsed: {time.time()-start} seconds (i.e. {str(datetime.timedelta(seconds = time.time()-start))})')
