"""Module with helper functions and classes
"""
import logging
import logging.config
import os
from pathlib import Path

PATH_TO_LOG_CONFIG = Path(os.getcwd()).parent / 'logging.conf'


def setup_logging(config_file: Path = PATH_TO_LOG_CONFIG):
    """Function to setup logging from file
    """
    logging.config.fileConfig(fname=config_file, disable_existing_loggers=False)
    # Get the logger specified in the file
    logger = logging.getLogger(__name__)
    logger.info('Logging configured')


