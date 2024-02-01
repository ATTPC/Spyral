import sys

sys.path.append("..")
import cProfile
from spyral.core.config import load_config
from spyral.run import run_spyral
from multiprocessing import SimpleQueue
from pathlib import Path

CONFIG_PATH: Path = Path("../local_config.json")


def profile_spyral():
    print("Running spyral profiling session...")
    print("Results will be written to prof.out...")
    print("Please be patient...")
    cProfile.run("run_profile()", "prof.out")
    print("Finished.")


def run_profile():
    config = load_config(CONFIG_PATH)
    runs = [config.run.run_min]
    queue = SimpleQueue()
    run_spyral(config, runs, queue, 1)


if __name__ == "__main__":
    profile_spyral()
