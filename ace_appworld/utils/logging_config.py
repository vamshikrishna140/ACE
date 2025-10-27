import logging
import sys
from ace_appworld import config

def setup_logging():
    """Configures the root logger based on settings in config.py."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)-22s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(config.LOG_LEVEL)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Console Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(config.LOG_LEVEL)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    
    # File Handler
    try:
        file_handler = logging.FileHandler(config.LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setLevel(config.LOG_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.error(f"Failed to create file handler at {config.LOG_FILE}: {e}")

    # Set third-party log levels higher to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Initialize logging immediately when this module is imported
setup_logging()

