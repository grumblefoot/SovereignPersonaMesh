import os
import logging
import logging.handlers
import pytest
import tempfile
from pathlib import Path
from proxy.core.logger import setup_rotating_logger

@pytest.fixture
def temp_log_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def test_setup_rotating_logger_creates_file(temp_log_dir):
    log_file = os.path.join(temp_log_dir, "test.log")
    
    logger = setup_rotating_logger(
        name="TestLogger1",
        level=logging.DEBUG,
        log_file=log_file
    )
    
    # Check if logger is configured properly
    assert logger.name == "TestLogger1"
    assert logger.level == logging.DEBUG
    
    # Should have two handlers: file and console
    assert len(logger.handlers) == 2
    
    file_handler = logger.handlers[0]
    console_handler = logger.handlers[1]
    
    assert isinstance(file_handler, logging.handlers.RotatingFileHandler)
    assert isinstance(console_handler, logging.StreamHandler)
    
    # Write a test log
    logger.info("Test message")
    
    # Verify file was created
    assert os.path.exists(log_file)
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "Test message" in content

def test_setup_rotating_logger_avoid_duplicates(temp_log_dir):
    log_file = os.path.join(temp_log_dir, "test2.log")
    
    logger1 = setup_rotating_logger(
        name="TestLogger2",
        level=logging.INFO,
        log_file=log_file
    )
    
    assert len(logger1.handlers) == 2
    
    # Call it again with the same name
    logger2 = setup_rotating_logger(
        name="TestLogger2",
        level=logging.INFO,
        log_file=log_file
    )
    
    # Should return the exact same logger without adding new handlers
    assert logger1 is logger2
    assert len(logger2.handlers) == 2

def test_setup_rotating_logger_creates_directory(temp_log_dir):
    # Pass a nested path that doesn't exist yet
    nested_dir = os.path.join(temp_log_dir, "nested", "logs")
    log_file = os.path.join(nested_dir, "test3.log")
    
    logger = setup_rotating_logger(
        name="TestLogger3",
        level=logging.INFO,
        log_file=log_file
    )
    
    logger.info("Directory test")
    
    assert os.path.exists(log_file)
    assert os.path.isdir(nested_dir)
