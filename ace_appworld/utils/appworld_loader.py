import logging
from typing import List
from ace_appworld import config

logger = logging.getLogger(__name__)

def load_appworld_tasks(split: str = config.APPWORLD_DATA_SPLIT) -> List[str]:
    """
    Loads the task IDs for the specified AppWorld data split.
    """
    try:
        from appworld import load_task_ids
    except ImportError:
        logger.error("AppWorld package not found. Run: uv pip install appworld")
        raise

    try:
        logger.info(f"Loading AppWorld task IDs for split: '{split}'...")
        task_ids = load_task_ids(split)
        if not task_ids:
            logger.error(f"No task IDs found for split '{split}'. Did you run 'appworld download data'?")
            raise ValueError(f"No tasks found for split '{split}'")
            
        logger.info(f"Loaded {len(task_ids)} task IDs for split '{split}'.")
        return task_ids
    except Exception as e:
        logger.error(f"Failed to load AppWorld tasks: {e}")
        logger.error("Please ensure AppWorld is installed and data is downloaded:")
        logger.error("  uv pip install appworld")
        logger.error("  appworld install")
        logger.error("  appworld download data")
        raise

