from pcutils.phase_1 import phase_1
from pcutils.core.config import load_config

def main():

    config = load_config('config.json')
    print("Starting Phase 1...")
    phase_1(config)
    print("Phase 1 complete.")

if __name__ == "__main__":
    main()