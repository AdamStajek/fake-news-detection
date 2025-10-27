import logging


def get_logger() -> logging.Logger:
    """Configure and return a logger instance.

    The logger is configured to write messages of level DEBUG and higher
    to a file named 'project.log' in the 'logs' directory.

    Args:
        name (str): The name for the logger, typically __name__.

    Returns:
        logging.Logger: The configured logger instance.

    """
    logger = logging.getLogger("project_logger")
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        log_file = "agents/logger/project.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

