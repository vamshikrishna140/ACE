import logging
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
from ace_appworld.utils.appworld_loader import load_appworld_tasks

def main():
    """
    Runs the full SEQUENTIAL OFFLINE adaptation pipeline (Algorithm 1).
    
    For each task in the training set:
    1. Generate (Attempt task with current playbook)
    2. Reflect (Analyze the trace, success or fail)
    3. Curate (Update the playbook with new insights)
    
    The playbook is updated *after each task*, so the agent
    gets progressively smarter for the next task.
    """
    logger.info("======== STARTING SEQUENTIAL OFFLINE ADAPTATION (Algorithm 1) ========")
    
    # 1. Initialize components
    logger.info("Initializing components (Agent, Reflector, Curator)...")
    curator = Curator()
    manager = PlaybookManager(curator)
    reflector = Reflector()
    
    # We initialize the agent once. We will manually update
    # its internal playbook representation before each task.
    agent = ReActAgent(playbook_path=config.PLAYBOOK_PATH, use_playbook=True)
    
    # 2. Load the training dataset
    # (Controlled by APPWORLD_DATA_SPLIT in config.py)
    try:
        task_list = load_appworld_tasks()
    except Exception:
        logger.error("Failed to load tasks. Exiting.")
        return

    # 3. Loop through all tasks
    for i, task_id in enumerate(task_list):
        logger.info(f"\n{'='*80}")
        logger.info(f"--- Processing Task {i+1}/{len(task_list)}: {task_id} ---")
        logger.info(f"{'='*80}")
        
        # --- 3a. GENERATE (Generate(t_i, P)) ---
        
        # Load the latest playbook and update the agent's in-memory copy
        current_playbook = manager.load_playbook(config.PLAYBOOK_PATH)
        agent.update_playbook(config.PLAYBOOK_PATH) 
        
        playbook_stats = manager.get_playbook_stats(current_playbook)
        logger.info(f"Running agent with playbook ({playbook_stats['total_bullets']} bullets)")
        
        episode = agent.run_episode(
            task_id=task_id,
            experiment_name=f"offline_run"
        )
        
        # Save the trace (episode)
        prefix = "success" if episode.success else "failure"
        episode_file = config.TRACE_OUTPUT_DIR / f"{prefix}_{task_id}.json"
        try:
            # default=lambda o: o.__dict__ helps serialize dataclasses
            with open(episode_file, 'w', encoding='utf-8') as f:
                json.dump(episode.__dict__, f, default=lambda o: o.__dict__, indent=2, ensure_ascii=False)
            logger.info(f"Trace saved to {episode_file}")
        except Exception as e:
            logger.error(f"Failed to save trace for {task_id}: {e}")

        # --- 3b. REFLECT (Reflect(tau_i)) ---
        if not episode.steps:
             logger.error(f"Episode for {task_id} had no steps. Cannot reflect. Skipping.")
             continue

        logger.info(f"Reflecting on episode for task {task_id}...")
        try:
            ground_truth = agent._load_ground_truth(task_id)
            exec_feedback = (
                f"Task {'succeeded' if episode.success else 'failed'}.\n"
                f"Error: {episode.error}\n"
                f"Final Answer: {episode.final_answer}"
            )
            
            # Get current playbook bullets for tagging
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
                execution_feedback=exec_feedback,
                playbook_bullets=playbook_bullets
            )
            
            # Save the reflection
            reflection_file = config.REFLECTION_OUTPUT_DIR / f"reflection_{task_id}.json"
            with open(reflection_file, 'w', encoding='utf-8') as f:
                json.dump(reflection.__dict__, f, default=lambda o: o.__dict__, indent=2, ensure_ascii=False)
            logger.info(f"Reflection saved to {reflection_file}")

        except Exception as e:
            logger.error(f"Reflection failed for {task_id}: {e}. Skipping curation.", exc_info=True)
            continue # Skip to the next task

        # --- 3c. CURATE (P <- Curate(P, r_i)) ---
        logger.info(f"Curating playbook from reflection for task {task_id}...")
        try:
            curation_result = curator.curate(
                task_context=episode.instruction,
                current_playbook=current_playbook,
                reflection=reflection
            )
            
            new_ops = curator.deduplicate_operations(curation_result.operations, current_playbook)
            if new_ops:
                current_playbook = manager.apply_operations(current_playbook, new_ops)

            if reflection.bullet_tags:
                current_playbook = manager.update_bullet_tags(current_playbook, reflection.bullet_tags)
            
            # Prune *every* time to keep playbook clean
            current_playbook = manager.prune_harmful_bullets(current_playbook)

            # Save the updated playbook *before the next task*
            manager.save_playbook(current_playbook, config.PLAYBOOK_PATH)
            logger.info(f"Playbook updated and saved. Ready for next task.")

        except Exception as e:
            logger.error(f"Curation failed for {task_id}: {e}. Playbook not updated.", exc_info=True)
            # We still continue to the next task, just with the old playbook
    
    logger.info("\n======== SEQUENTIAL OFFLINE ADAPTATION COMPLETE ========")
    stats = manager.get_playbook_stats(current_playbook)
    logger.info(f"Final playbook contains {stats['total_bullets']} bullets.")

if __name__ == "__main__":
    main()

