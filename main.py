from pcutils.core.config import load_config
from pcutils.run import run_pcutils
import sys

def main(config_path: str):
    config = load_config(config_path)
    run_pcutils(config)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('pcutils requires a configuration file!')
    else:
        main(sys.argv[1])