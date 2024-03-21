import logging

from sssd.utils.logger import setup_logger


def test_setup_logger():
    # Test if logger is configured correctly
    logger = setup_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO

    # Check if logger has a console handler
    assert any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    )

    # Test reusing the logger
    reused_logger = setup_logger()
    assert reused_logger is logger  # Should return the same logger instance

    # Test logger configuration after reusing
    assert reused_logger.level == logging.INFO
    assert any(
        isinstance(handler, logging.StreamHandler) for handler in reused_logger.handlers
    )

    # Test adding an additional file handler
    file_handler = logging.FileHandler("test.log")
    reused_logger.addHandler(file_handler)
    assert any(
        isinstance(handler, logging.FileHandler) for handler in reused_logger.handlers
    )
