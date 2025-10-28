import logging
import json
import argparse
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

def main(start_from=0):
    """
    Runs the full SEQUENTIAL OFFLINE adaptation pipeline with EPOCHS (Algorithm 1).
    
    For each epoch:
        For each task in the training set:
            1. Generate (Attempt task with current playbook)
            2. Reflect (Analyze the trace, success or fail)
            3. Curate (Update the playbook with new insights)
    
    Training continues for NUM_EPOCHS or until all tasks succeed.
    """
    logger.info("======== STARTING SEQUENTIAL OFFLINE ADAPTATION WITH EPOCHS (Algorithm 1) ========")
    
    # 1. Initialize components
    logger.info("Initializing components (Agent, Reflector, Curator)...")
    curator = Curator()
    manager = PlaybookManager(curator)
    reflector = Reflector()
    
    # Initialize agent once
    agent = ReActAgent(playbook_path=config.PLAYBOOK_PATH, use_playbook=True)
    
    # 2. Load the training dataset
    try:
        task_list = load_appworld_tasks()
    except Exception:
        logger.error("Failed to load tasks. Exiting.")
        return

    # Track successful tasks across epochs
    successful_tasks = set()
    
    # 3. Loop through epochs
    for epoch in range(config.NUM_EPOCHS):
        logger.info(f"\n{'#'*100}")
        logger.info(f"############ EPOCH {epoch + 1}/{config.NUM_EPOCHS} ############")
        logger.info(f"{'#'*100}")
        logger.info(f"Successful tasks so far: {len(successful_tasks)}/{len(task_list)}")
        
        # Check if all tasks are successful
        if len(successful_tasks) == len(task_list):
            logger.info("🎉🎉🎉 ALL TASKS SUCCESSFUL! Training complete. 🎉🎉🎉")
            break
        
        # Track success in current epoch
        epoch_successes = 0
        epoch_failures = 0
        
        # 4. Loop through all tasks
        for i, task_id in enumerate(task_list[start_from:], start=start_from):
            logger.info(f"\n{'='*80}")
            logger.info(f"--- Epoch {epoch + 1}, Task {i+1}/{len(task_list)}: {task_id} ---")
            if task_id in successful_tasks:
                logger.info(f"[ALREADY SUCCESSFUL - Skipping]")
                logger.info(f"{'='*80}")
                continue
            logger.info(f"{'='*80}")
            
            # --- 4a. GENERATE (Generate(t_i, P)) ---
            
            # Load the latest playbook and update the agent's in-memory copy
            current_playbook = manager.load_playbook(config.PLAYBOOK_PATH)
            agent.update_playbook(config.PLAYBOOK_PATH) 
            
            playbook_stats = manager.get_playbook_stats(current_playbook)
            logger.info(f"Running agent with playbook ({playbook_stats['total_bullets']} bullets)")
            
            episode = agent.run_episode(
                task_id=task_id,
                experiment_name=f"offline_epoch{epoch+1}_run_{task_id}"
            )
            
            # Save the trace (episode)
            prefix = "success" if episode.success else "failure"
            episode_file = config.TRACE_OUTPUT_DIR / f"epoch{epoch+1}_{prefix}_{task_id}.json"
            try:
                with open(episode_file, 'w', encoding='utf-8') as f:
                    json.dump(episode.__dict__, f, default=lambda o: o.__dict__, indent=2, ensure_ascii=False)
                logger.info(f"Trace saved to {episode_file}")
            except Exception as e:
                logger.error(f"Failed to save trace for {task_id}: {e}")

            # Track success
            if episode.success:
                successful_tasks.add(task_id)
                epoch_successes += 1
                logger.info(f"✓✓✓ Task {task_id} SUCCEEDED ✓✓✓")
            else:
                epoch_failures += 1
                logger.warning(f"✗✗✗ Task {task_id} FAILED ✗✗✗")

            # --- 4b. REFLECT (Reflect(tau_i)) ---
            if not episode.steps:
                 logger.error(f"Episode for {task_id} had no steps. Cannot reflect. Skipping.")
                 continue

            logger.info(f"Reflecting on episode for task {task_id}...")
            try:
                ground_truth = agent._load_ground_truth(task_id)

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
                    playbook_bullets=playbook_bullets,
                    evaluation_report=episode.evaluation_report
                )
                
                # Save the reflection
                reflection_file = config.REFLECTION_OUTPUT_DIR / f"epoch{epoch+1}_reflection_{task_id}.json"
                with open(reflection_file, 'w', encoding='utf-8') as f:
                    json.dump(reflection.__dict__, f, default=lambda o: o.__dict__, indent=2, ensure_ascii=False)
                logger.info(f"Reflection saved to {reflection_file}")

            except Exception as e:
                logger.error(f"Reflection failed for {task_id}: {e}. Skipping curation.", exc_info=True)
                continue

            # --- 4c. CURATE (P <- Curate(P, r_i)) ---
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
                
                # Prune every time to keep playbook clean
                current_playbook = manager.prune_harmful_bullets(current_playbook)

                # Save the updated playbook
                manager.save_playbook(current_playbook, config.PLAYBOOK_PATH)
                logger.info(f"Playbook updated and saved. Ready for next task.")

            except Exception as e:
                logger.error(f"Curation failed for {task_id}: {e}. Playbook not updated.", exc_info=True)
        
        # Epoch summary
        logger.info(f"\n{'#'*100}")
        logger.info(f"EPOCH {epoch + 1} SUMMARY:")
        logger.info(f"  Successes: {epoch_successes}")
        logger.info(f"  Failures: {epoch_failures}")
        logger.info(f"  Total Successful Tasks: {len(successful_tasks)}/{len(task_list)}")
        logger.info(f"  Success Rate: {len(successful_tasks)/len(task_list)*100:.2f}%")
        logger.info(f"{'#'*100}\n")
    
    # Final summary
    logger.info("\n======== SEQUENTIAL OFFLINE ADAPTATION COMPLETE ========")
    stats = manager.get_playbook_stats(current_playbook)
    logger.info(f"Final playbook contains {stats['total_bullets']} bullets.")
    logger.info(f"Total successful tasks: {len(successful_tasks)}/{len(task_list)}")
    logger.info(f"Final success rate: {len(successful_tasks)/len(task_list)*100:.2f}%")
    
    if len(successful_tasks) == len(task_list):
        logger.info("🎉🎉🎉 ALL TASKS COMPLETED SUCCESSFULLY! 🎉🎉🎉")
    else:
        failed_tasks = set(task_list) - successful_tasks
        logger.info(f"Failed tasks: {failed_tasks}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run sequential offline adaptation with epochs')
    parser.add_argument(
        '--start_from',
        type=int,
        default=0,
        help='Index of the task to start from (0-based, default: 0)'
    )
    
    args = parser.parse_args()
    main(start_from=args.start_from)