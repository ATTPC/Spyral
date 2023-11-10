from spyral.core.config import load_config
from spyral.run_parallel import run_spyral_parallel
import sys
from pathlib import Path

SPLASH: str = \
r"""
-------------------------------
 ____                        _ 
/ ___| _ __  _   _ _ __ __ _| |
\___ \|  _ \| | | |  __/ _  | |
 ___| | |_| | |_| | | | |_| | |
|____/|  __/ \__  |_|  \__ _|_|
      |_|    |___/             
-------------------------------
"""


def show_splash():
    print(SPLASH)

def main(config_path: str):
    show_splash()
    config = load_config(Path(config_path))
    run_spyral_parallel(config)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('spyral requires a configuration file!')
    else:
        main(sys.argv[1])
