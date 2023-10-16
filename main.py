from spyral.core.config import load_config
from spyral.run import run_spyral
import sys

def main(config_path: str):
    config = load_config(config_path)
    run_spyral(config)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('spyral requires a configuration file!')
    else:
        main(sys.argv[1])
