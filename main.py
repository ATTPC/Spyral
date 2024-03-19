from spyral.core.config import load_config
from spyral.run_parallel import run_spyral_parallel

from pathlib import Path
import click
import contextlib
import os

# Generated using https://www.asciiart.eu
SPLASH: str = r"""
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


@click.command()
@click.option(
    "--term/--no-term",
    default=True,
    help="Whether or not Spyral displays progress text to the terminal",
    show_default=True,
)
@click.argument("config", type=click.Path(exists=True))
def main(term: bool, config: str):
    """
    Spyral is an analysis framework for AT-TPC data. Provide a JSON configuration file CONFIG to control analysis settings.
    """
    configuration = load_config(Path(config))
    if not term:
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            run_spyral_parallel(configuration, no_progress=True)
    else:
        show_splash()
        run_spyral_parallel(configuration)


if __name__ == "__main__":
    main()
