from spyral.core.config import load_config
# from spyral.run import run_spyral
from spyral.run_parallel import run_spyral_parallel
import sys
from pathlib import Path

def main(config_path: str):
    config = load_config(Path(config_path))
    run_spyral_parallel(config)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('spyral requires a configuration file!')
    else:
        main(sys.argv[1])
