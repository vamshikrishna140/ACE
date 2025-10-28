import logging
import sys
import json
from ace_appworld.utils.logging_config import setup_logging
from ace_appworld import config

setup_logging()
logger = logging.getLogger(__name__)

from ace_appworld.components.agent import ReActAgent
from ace_appworld.components.reflector import Reflector
from ace_appworld.components.curator import Curator 
from ace_appworld.components.playbook import PlaybookManager
from ace_appworld.components.thompson_sampling import ThompsonSamplingPolicy
from ace_appworld.components.online_reward_tracker import OnlineRewardTracker

def run_online_loop_with_thompson_sampling(task_id: str, thompson_policy=None):
    """
    Enhanced online loop with Thompson Sampling for retry decisions.
    
    Key improvements over heuristic approach:
    - Learns component reliability from experience
    - Balances exploration vs exploitation
    - Provides principled credit attribution
    - Converges to optimal retry policy
    
    Args:
        task_id: Task identifier
        thompson_policy: Optional shared policy for batch learning
    """
    logger.info(f"{'='*80}")
    logger.info(f"ONLINE ADAPTATION WITH THOMPSON SAMPLING: {task_id}")
    logger.info(f"{'='*80}")
    
    # Initialize components
    curator = Curator()
    manager = PlaybookManager(curator)
    reflector = Reflector()
    agent = ReActAgent(playbook_path=config.PLAYBOOK_PATH, use_playbook=True)
    
    # Initialize or use provided Thompson Sampling policy
    if thompson_policy is None:
        thompson_policy = ThompsonSamplingPolicy(
            attempt_rewards={1: 1.0, 2: 0.6, 3: 0.3},
            cost_per_attempt=0.1,
            use_quality_weighted_attribution=True
        )
    
    reward_tracker = OnlineRewardTracker(thompson_policy=thompson_policy)
    
    cumulative_reward = 0.0
    reflection_quality = 0.0
    curation_value = 0.0
    
    for attempt in range(1, config.MAX_ONLINE_RETRIES + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"ATTEMPT {attempt}/{config.MAX_ONLINE_RETRIES}")
        logger.info(f"{'='*80}")
        
        # Load playbook
        current_playbook = manager.load_playbook(config.PLAYBOOK_PATH)
        agent.update_playbook(config.PLAYBOOK_PATH)
        
        playbook_stats = manager.get_playbook_stats(current_playbook)
        logger.info(f"Playbook: {playbook_stats['total_bullets']} bullets")
        
        # === ATTEMPT TASK ===
        episode = agent.run_episode(
            task_id=task_id,
            experiment_name=f"online_attempt{attempt}_{task_id}"
        )
        
        # Calculate reward for this attempt
        attempt_reward = reward_tracker.calculate_attempt_reward(episode, attempt)
        cumulative_reward += attempt_reward
        
        logger.info(f"Attempt Reward: {attempt_reward:+.2f} (Cumulative: {cumulative_reward:+.2f})")
        
        # === CHECK SUCCESS ===
        if episode.success:
            logger.info(f"✓✓✓ SUCCESS on attempt {attempt}! ✓✓✓")
            
            # Update Thompson Sampling beliefs with success
            reward_tracker.update_from_outcome(
                success=True,
                attempt_num=attempt,
                reflection_quality=reflection_quality,
                curation_value=curation_value
            )
            
            return True, cumulative_reward
        
        # === HANDLE FAILURE ===
        logger.warning(f"✗ Attempt {attempt} failed.")
        
        if attempt == config.MAX_ONLINE_RETRIES:
            logger.error(f"Max retries reached. Task failed.")
            
            # Update with failure
            reward_tracker.update_from_outcome(
                success=False,
                attempt_num=attempt,
                reflection_quality=reflection_quality,
                curation_value=curation_value
            )
            
            return False, cumulative_reward
        
        # === REFLECTION ===
        if not episode.steps:
            logger.error("No steps to reflect on. Aborting.")
            reward_tracker.update_from_outcome(False, attempt, 0.0, 0.0)
            return False, cumulative_reward
        
        logger.info(f"\n{'─'*60}")
        logger.info("REFLECTION PHASE")
        logger.info(f"{'─'*60}")
        
        try:
            ground_truth = agent._load_ground_truth(task_id)
            
            playbook_bullets = []
            for section, bullets in current_playbook.items():
                if isinstance(bullets, list):
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
            
            # Assess reflection quality
            reflection_quality = reward_tracker.calculate_reflection_quality(
                reflection, episode
            )
            
            logger.info(f"Reflection Quality Score: {reflection_quality:.2f}")
            logger.info(f"Key Insight: {reflection.key_insight[:100]}...")
            
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            reward_tracker.update_from_outcome(False, attempt, 0.0, 0.0)
            return False, cumulative_reward
        
        # === CURATION ===
        logger.info(f"\n{'─'*60}")
        logger.info("CURATION PHASE")
        logger.info(f"{'─'*60}")
        
        try:
            curation_result = curator.curate(
                task_context=episode.instruction,
                current_playbook=current_playbook,
                reflection=reflection
            )
            
            # Assess curation value
            curation_value = reward_tracker.calculate_curation_value(
                curation_result, current_playbook
            )
            
            logger.info(f"Curation Value Score: {curation_value:.2f}")
            logger.info(f"Proposed Operations: {len(curation_result.operations)}")
            
            # === THOMPSON SAMPLING DECISION ===
            should_retry, reason = reward_tracker.should_retry(
                attempt, reflection_quality, curation_value
            )
            
            logger.info(f"\n{'*'*70}")
            logger.info(f"THOMPSON SAMPLING DECISION: {'✓ RETRY' if should_retry else '✗ SKIP'}")
            logger.info(f"Reason: {reason}")
            logger.info(f"{'*'*70}\n")
            
            if not should_retry:
                logger.warning(f"Thompson Sampling suggests skipping retry. Moving to next task.")
                reward_tracker.update_from_outcome(
                    False, attempt, reflection_quality, curation_value
                )
                return False, cumulative_reward
            
            # Apply curation if we decided to retry
            new_ops = curator.deduplicate_operations(
                curation_result.operations, current_playbook
            )
            
            if new_ops:
                current_playbook = manager.apply_operations(current_playbook, new_ops)
                logger.info(f"Applied {len(new_ops)} new operations")
            else:
                logger.warning("No new operations after deduplication!")
            
            if reflection.bullet_tags:
                current_playbook = manager.update_bullet_tags(
                    current_playbook, reflection.bullet_tags
                )
            
            manager.save_playbook(current_playbook, config.PLAYBOOK_PATH)
            
            logger.info("Playbook updated. Preparing for retry...")
            
        except Exception as e:
            logger.error(f"Curation failed: {e}")
            reward_tracker.update_from_outcome(
                False, attempt, reflection_quality, 0.0
            )
            return False, cumulative_reward
    
    # Should never reach here
    return False, cumulative_reward


def run_batch_online_adaptation(task_ids: list):
    """
    Run Thompson Sampling across multiple tasks to learn optimal policy.
    
    This is where Thompson Sampling really shines - it learns from
    experience across tasks and improves retry decisions over time.
    
    Args:
        task_ids: List of task identifiers to process
        
    Returns:
        Dictionary containing results and learning statistics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"BATCH ONLINE ADAPTATION: {len(task_ids)} tasks")
    logger.info(f"{'='*80}\n")
    
    # Shared Thompson Sampling policy (learns across tasks)
    thompson_policy = ThompsonSamplingPolicy(
        attempt_rewards={1: 1.0, 2: 0.6, 3: 0.3},
        cost_per_attempt=0.1,
        use_quality_weighted_attribution=True
    )
    
    results = []
    total_reward = 0.0
    successes = 0
    
    for i, task_id in enumerate(task_ids, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"TASK {i}/{len(task_ids)}: {task_id}")
        logger.info(f"{'#'*80}")
        
        try:
            # Run task with shared policy (learns across tasks)
            success, task_reward = run_online_loop_with_thompson_sampling(
                task_id, thompson_policy=thompson_policy
            )
            
            results.append({
                'task_id': task_id,
                'success': success,
                'reward': task_reward
            })
            
            total_reward += task_reward
            if success:
                successes += 1
            
            logger.info(f"\nTask Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
            logger.info(f"Task Reward: {task_reward:+.2f}")
            logger.info(f"Running Totals: {successes}/{i} successes, {total_reward:+.2f} reward")
            
            # Print learning progress every 5 tasks
            if i % 5 == 0:
                logger.info(f"\n{'─'*70}")
                logger.info(f"LEARNING PROGRESS AFTER {i} TASKS")
                logger.info(f"{'─'*70}")
                thompson_policy.print_summary()
                
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            results.append({
                'task_id': task_id,
                'success': False,
                'reward': 0.0,
                'error': str(e)
            })
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"BATCH ADAPTATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total Tasks: {len(task_ids)}")
    logger.info(f"Successes: {successes}/{len(task_ids)} ({successes/len(task_ids)*100:.1f}%)")
    logger.info(f"Total Reward: {total_reward:+.2f}")
    logger.info(f"Average Reward: {total_reward/len(task_ids):+.2f}")
    logger.info(f"{'='*80}\n")
    
    # Print final learned policy
    thompson_policy.print_summary()
    
    return {
        'results': results,
        'total_reward': total_reward,
        'success_rate': successes / len(task_ids),
        'policy_stats': thompson_policy.get_component_stats()
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Main entry point with Thompson Sampling"""
    if len(sys.argv) < 2:
        print("Usage: python run_online_thompson.py <task_id> [task_id2 ...]")
        print("   or: python run_online_thompson.py --batch <task_list_file>")
        print("\nRuns online adaptation with Thompson Sampling for retry decisions.")
        print("\nExamples:")
        print("  Single task:")
        print("    python run_online_thompson.py b0a8eae_3")
        print("\n  Multiple tasks:")
        print("    python run_online_thompson.py task1 task2 task3")
        print("\n  Batch from file:")
        print("    python run_online_thompson.py --batch tasks.txt")
        sys.exit(1)
    
    # Parse arguments
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            logger.error("--batch requires a task list file")
            sys.exit(1)
        
        # Load tasks from file
        task_file = sys.argv[2]
        try:
            with open(task_file, 'r') as f:
                task_ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(task_ids)} tasks from {task_file}")
        except Exception as e:
            logger.error(f"Failed to load task file: {e}")
            sys.exit(1)
        
        # Run batch adaptation
        batch_results = run_batch_online_adaptation(task_ids)
        
        # Save results
        output_file = "thompson_batch_results.json"
        with open(output_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
        
    else:
        # Single or multiple tasks from command line
        task_ids = sys.argv[1:]
        
        if len(task_ids) == 1:
            # Single task mode
            task_id = task_ids[0]
            
            # Validate task
            task_path = config.APPWORLD_DATA_DIR / "tasks" / task_id
            if not task_path.exists():
                logger.error(f"Task '{task_id}' not found at {task_path}")
                sys.exit(1)
            
            # Run single task
            success, total_reward = run_online_loop_with_thompson_sampling(task_id)
            
            # Final summary
            logger.info(f"\n{'='*80}")
            logger.info(f"FINAL RESULT: {'SUCCESS ✓' if success else 'FAILED ✗'}")
            logger.info(f"Total Reward: {total_reward:+.2f}")
            logger.info(f"{'='*80}")
            
            return success
        else:
            # Multiple tasks - use batch mode
            logger.info(f"Running {len(task_ids)} tasks in batch mode")
            batch_results = run_batch_online_adaptation(task_ids)
            
            # Save results
            output_file = "thompson_batch_results.json"
            with open(output_file, 'w') as f:
                json.dump(batch_results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
            
            return batch_results['success_rate'] > 0.5


if __name__ == "__main__":
    main()