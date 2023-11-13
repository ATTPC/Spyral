import logging
from .workspace import Workspace
from typing import Callable
import functools

ROOT_LOGGER = 'spyral'

def init_spyral_logger_parent(ws: Workspace):
    logger = logging.getLogger(ROOT_LOGGER)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(ws.get_log_file_path(-1))
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)

def init_spyral_logger_child(ws: Workspace, process_id: int):
    '''
    Setup the spyral logging system. Should only be called *once* at the start of
    each child process.
    
    This will create a log file in the log/ directory of the workspace, 
    where the name indicates which child process is logging to it. All stdout messages are then written to
    this log file. Each process gets its own log to avoid any need for synchronization.

    ## Parameters
    ws: Workspace, the project workspace
    process_id: int, the process id used to name the log file
    '''
    logger = logging.getLogger(ROOT_LOGGER)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(ws.get_log_file_path(process_id))
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)

def get_spyral_logger(module: str) -> logging.Logger:
    '''
    Get the spyral logger, decorating it with the module name from which it is being called,
    allowing for easier tracing of errors. Can 

    '''
    return logging.getLogger(module)

def spyral_error(module: str, message: str):
    logger = get_spyral_logger(module)
    logger.error(message)

def spyral_warn(module: str, message: str):
    logger = get_spyral_logger(module)
    logger.warn(message)

def spyral_info(module: str, message: str):
    logger = get_spyral_logger(module)
    logger.info(message)

def spyral_except(module: str, exception: Exception):
    logger = get_spyral_logger(module)
    logger.exception(exception)

def spyral_debug(module: str, message: str):
    logger = get_spyral_logger(module)
    logger.debug(message)