from pcutils.core.config import load_config
from pcutils.run import run_pcutils

def main():
    config = load_config('config.json')
    run_pcutils(config)

if __name__ == "__main__":
    main()