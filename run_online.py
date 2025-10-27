import logging
import sys
import json
from ace_appworld.utils.logging_config import setup_logging
from ace_appworld import config

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

from ace_appworld.components.agent import ReActAgent
from ace_appworld.components.reflector import Reflector
from ace_appworld.components.curator import Curator 
from ace_appworld.components.playbook import PlaybookManager
from ace_appworld.components.models import Episode

def run_online_loop(task_id: str):
    """
    Runs the ONLINE adaptation loop for a single task (Algorithm 2).
    
    It will attempt the task, and if it fails, it will reflect,
    update the playbook, and retry until success or max retries.
    """
    logger.info(f"======== STARTING ONLINE ADAPTATION (Algorithm 2) FOR TASK: {task_id} ========")
    
    # 1. Initialize components
    curator = Curator()
    manager = PlaybookManager(curator)
    reflector = Reflector()
    
    # Instantiate agent once, we will update its playbook if needed
    agent = ReActAgent(playbook_path=config.PLAYBOOK_PATH, use_playbook=True, enable_validation=True)
    
    for attempt in range(config.MAX_ONLINE_RETRIES):
        logger.info(f"--- Attempt {attempt + 1}/{config.MAX_ONLINE_RETRIES} for Task {task_id} ---")
        
        # 1. Load the most recent playbook
        current_playbook = manager.load_playbook(config.PLAYBOOK_PATH)
        playbook_stats = manager.get_playbook_stats(current_playbook)
        logger.info(f"Loaded playbook with {playbook_stats['total_bullets']} bullets")
        
        # Update agent's in-memory playbook (in case it changed)
        agent.update_playbook(config.PLAYBOOK_PATH)
        
        # 2. Run the Agent (Generate(t, P))
        episode = agent.run_episode(
            task_id=task_id,
            experiment_name=f"online_run"
        )
        
        # 3. Check for Success
        if episode.success:
            logger.info(f"✓✓✓ Task {task_id} SUCCEEDED on attempt {attempt + 1}. ✓✓✓")
            logger.info(f"======== ONLINE ADAPTATION FOR {task_id} COMPLETE ========")
            return True # Task succeeded

        # 4. Handle Failure (Reflect & Curate)
        logger.warning(f"Task {task_id} FAILED on attempt {attempt + 1}. Starting reflection...")
        if episode.error:
            logger.warning(f"Episode Error: {episode.error}")

        if not episode.steps:
            logger.error("Episode failed with no steps. Cannot reflect. Stopping.")
            break

        # --- Reflection (Reflect(tau_k)) ---
        try:
            ground_truth = agent._load_ground_truth(task_id) 
            execution_feedback = (
                f"Task failed on attempt {attempt + 1} with error: {episode.error}\n"
                f"Final Answer: {episode.final_answer}"
            )
            
            # Load bullets for tagging
            playbook_bullets = []
            for section, bullets in current_playbook.items():
                if not isinstance(bullets, list): continue
                for bullet in bullets:
                    bullet['section'] = section
                    playbook_bullets.append(bullet)
            
            reflection = reflector.reflect(
                task_instruction=episode.instruction,
                trajectory=[s.__dict__ for s in episode.steps],
                final_answer=episode.final_answer,
                ground_truth=ground_truth,
                execution_feedback=execution_feedback,
                playbook_bullets=playbook_bullets
            )
        except Exception as e:
            logger.error(f"Reflection failed: {e}. Stopping retries for this task.", exc_info=True)
            break

        # --- Curation (P <- Curate(P, r_k)) ---
        try:
            # a. Curate new insights
            curation_result = curator.curate(
                task_context=episode.instruction,
                current_playbook=current_playbook,
                reflection=reflection
            )
            
            # b. Deduplicate and apply new operations
            new_ops = curator.deduplicate_operations(curation_result.operations, current_playbook)
            if new_ops:
                current_playbook = manager.apply_operations(current_playbook, new_ops)
                logger.info(f"Applied {len(new_ops)} new operations to playbook.")

            # c. Update bullet tags
            if reflection.bullet_tags:
                current_playbook = manager.update_bullet_tags(current_playbook, reflection.bullet_tags)
            
            # d. Save the updated playbook *immediately*
            manager.save_playbook(current_playbook, config.PLAYBOOK_PATH)
            logger.info("Playbook updated. Retrying task...")

        except Exception as e:
            logger.error(f"Curation failed: {e}. Stopping retries for this task.", exc_info=True)
            break

    # 5. Max Retries Reached
    logger.error(f"✗✗✗ Task {task_id} FAILED after {config.MAX_ONLINE_RETRIES} attempts. ✗✗✗")
    logger.info(f"======== ONLINE ADAPTATION FOR {task_id} FAILED ========")
    return False

def main():
    """
    Main entry point for the online runner.
    Expects a single task_id as a command-line argument.
    """
    if len(sys.argv) < 2:
        print("Usage: python run_online.py <task_id>")
        print("\nThis script runs a single task with an online 'retry-on-failure' loop (Algorithm 2).")
        print("Example:")
        print("  python run_online.py b0a8eae_3")
        sys.exit(1)
        
    task_id = sys.argv[1]
    
    # Check if task_id is valid
    task_path = config.APPWORLD_DATA_DIR / "tasks" / task_id
    if not task_path.exists():
        logger.error(f"Task ID '{task_id}' not found at {task_path}")
        logger.error("Please make sure you've downloaded the AppWorld data.")
        sys.exit(1)
    
    run_online_loop(task_id)

if __name__ == "__main__":
    main()

