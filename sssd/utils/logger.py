import logging


def setup_logger() -> logging.Logger:
    # Check if logger has already been configured
    if logging.getLogger(__name__).handlers:
        return logging.getLogger(__name__)

    # Create a new logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and set it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    return logger
