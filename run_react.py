import logging
import sys
from ace_appworld.utils.logging_config import setup_logging
from ace_appworld import config

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

from ace_appworld.components.agent import ReActAgent

def run_single_task(task_id: str):
    """
    Runs the ReAct agent once for a single task without any adaptation loop.
    """
    logger.info(f"======== RUNNING TASK: {task_id} ========")
    
    # Initialize agent
    agent = ReActAgent(
        playbook_path=config.PLAYBOOK_PATH, 
        use_playbook=False, 
        enable_validation=True
    )
    
    # Run the agent once
    logger.info(f"Running agent for task {task_id}...")
    episode = agent.run_episode(
        task_id=task_id,
        experiment_name=f"react_run"
    )
    
    # Check result
    if episode.success:
        logger.info(f"✓✓✓ Task {task_id} SUCCEEDED ✓✓✓")
        logger.info(f"Final Answer: {episode.final_answer}")
        logger.info(f"======== TASK {task_id} COMPLETE ========")
        return True
    else:
        logger.error(f"✗✗✗ Task {task_id} FAILED ✗✗✗")
        if episode.error:
            logger.error(f"Error: {episode.error}")
        logger.error(f"Final Answer: {episode.final_answer}")
        logger.info(f"======== TASK {task_id} COMPLETE ========")
        return False

def main():
    """
    Main entry point for the simplified online runner.
    Expects a single task_id as a command-line argument.
    """
    if len(sys.argv) < 2:
        print("Usage: python run_react.py <task_id>")
        print("\nThis script runs a single task with the ReAct agent once.")
        print("Example:")
        print("  python run_react.py b0a8eae_3")
        sys.exit(1)
        
    task_id = sys.argv[1]
    
    # Check if task_id is valid
    task_path = config.APPWORLD_DATA_DIR / "tasks" / task_id
    if not task_path.exists():
        logger.error(f"Task ID '{task_id}' not found at {task_path}")
        logger.error("Please make sure you've downloaded the AppWorld data.")
        sys.exit(1)
    
    run_single_task(task_id)

if __name__ == "__main__":
    main()