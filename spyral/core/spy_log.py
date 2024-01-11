from .workspace import Workspace

import logging

ROOT_LOGGER = "spyral"


def init_spyral_logger_parent(ws: Workspace):
    """Setup the Spyral logger for the parent process

    Should only be used by the parent process

    Parameters
    ----------
    ws: Workspace
        The project Workspace
    """
    logger = logging.getLogger(ROOT_LOGGER)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(ws.get_log_file_path(-1))
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)


def init_spyral_logger_child(ws: Workspace, process_id: int):
    """Setup the spyral logging system for a child process.

    Should only be called *once* at the start of each child process.

    This will create a log file in the log/ directory of the workspace,
    where the name indicates which child process is logging to it. All stdout messages are then written to
    this log file. Each process gets its own log to avoid any need for synchronization.

    Parameters
    ----------
    ws: Workspace
        The project Workspace
    process_id: int
        The process id used to name the log file
    """
    logger = logging.getLogger(ROOT_LOGGER)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(ws.get_log_file_path(process_id))
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)


def get_spyral_logger(module: str) -> logging.Logger:
    """Get the spyral logger for this process,

    The logger is decorated with the module name from which it is being called,
    allowing for easier tracing of errors.

    Do not call this directly, prefer one of the spyral_* log functions

    Parameters
    ----------
    module: str
        The module name from which the logger is called

    Returns
    -------
    logging.Logger
        The Logger

    """
    return logging.getLogger(module)


def spyral_error(module: str, message: str):
    """Log an error

    Use like

    ```python
    spyral_error(__name__, "There was an error")
    ```
    The magic `__name__` variable will be the module name

    Parameters
    ----------
    module: str
        The module from which the logger was called
    message: str
        The log message
    """
    logger = get_spyral_logger(module)
    logger.error(message)


def spyral_warn(module: str, message: str):
    """Log a warning

    Use like

    ```python
    spyral_warn(__name__, "There was a warning")
    ```
    The magic `__name__` variable will be the module name

    Parameters
    ----------
    module: str
        The module from which the logger was called
    message: str
        The log message
    """
    logger = get_spyral_logger(module)
    logger.warn(message)


def spyral_info(module: str, message: str):
    """Log some info

    Use like

    ```python
    spyral_info(__name__, "Here's some info")
    ```
    The magic `__name__` variable will be the module name

    Parameters
    ----------
    module: str
        The module from which the logger was called
    message: str
        The log message
    """
    logger = get_spyral_logger(module)
    logger.info(message)


def spyral_except(module: str, exception: Exception):
    """Log an exception

    Use like

    ```python
    try:
        ...
    except Exception as e
        spyral_except(__name__, e)
    ```
    The magic `__name__` variable will be the module name

    Parameters
    ----------
    module: str
        The module from which the logger was called
    exception: Exception
        The exception to be logged
    """
    logger = get_spyral_logger(module)
    logger.exception(exception)


def spyral_debug(module: str, message: str):
    """Log a debug message

    Use like

    ```python
    spyral_debug(__name__, "Here's some debug info")
    ```
    The magic `__name__` variable will be the module name

    Parameters
    ----------
    module: str
        The module from which the logger was called
    message: str
        The log message
    """
    logger = get_spyral_logger(module)
    logger.debug(message)
