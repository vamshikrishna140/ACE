import json
import os
import re
import sys
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from ace_appworld.components.models import Episode, Step
from ace_appworld.components.prompts import get_generator_prompt
from ace_appworld.components.llm import LLM
from ace_appworld import config

# Setup logger
logger = logging.getLogger(__name__)

class ReActAgent:
    """
    ReAct Agent (Generator) for AppWorld Environment
    Implements the Reason-Act-Observe loop using AppWorld's built-in evaluation.
    """
    
    def __init__(self, 
             playbook_path: str = config.PLAYBOOK_PATH,
             use_playbook: bool = True,
             max_playbook_bullets: Optional[int] = None):
        
        self.data_dir = config.APPWORLD_DATA_DIR
        self.model_provider = config.GENERATOR_PROVIDER
        self.model_name = config.GENERATOR_MODEL
        self.max_steps = config.MAX_EPISODE_STEPS
        self.use_playbook = use_playbook
        self.max_playbook_bullets = max_playbook_bullets
        self.model_provider = config.GENERATOR_PROVIDER

        self.llm = LLM(model_name=self.model_name, model_provider=self.model_provider)

        # Load domain knowledge
        self.playbook = self._load_playbook(playbook_path) if use_playbook else ""
        
        # Verify dependencies
        self._check_appworld()
        
        logger.info(f"ReAct Agent initialized: Provider={self.model_provider}, Model={self.model_name}")
    
    def update_playbook(self, playbook_path: str):
        """Public method to reload the playbook from disk."""
        logger.info("Agent is reloading playbook...")
        self.playbook = self._load_playbook(playbook_path)
        
    def _load_playbook(self, playbook_path: str) -> str:
        """Load and format domain knowledge playbook"""
        if not os.path.exists(playbook_path):
            logger.warning(f"Playbook file not found: {playbook_path}. Using empty playbook.")
            return ""
        
        try:
            with open(playbook_path, 'r') as f:
                playbook_data = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {playbook_path}. Using empty playbook.")
            return ""
        except Exception as e:
            logger.error(f"Failed to load playbook {playbook_path}: {e}. Using empty playbook.")
            return ""

        playbook_text = ""
        bullet_count = 0
        
        for section, bullets in playbook_data.items():
            if not bullets or not isinstance(bullets, list):
                continue
            
            section_header = f"\n## {section.replace('_', ' ').title()}\n\n"
            section_content = []
            
            for bullet in bullets:
                if self.max_playbook_bullets and bullet_count >= self.max_playbook_bullets:
                    break
                
                content = bullet.get('content', '')
                if not content:
                    continue

                helpful = bullet.get('helpful', 0)
                harmful = bullet.get('harmful', 0)
                bullet_id = bullet.get('id', '')
                
                section_content.append(
                    f"- [{bullet_id}] (helpful={helpful}, harmful={harmful}): {content}\n"
                )
                bullet_count += 1
            
            if section_content:
                playbook_text += section_header + ''.join(section_content)
            
            if self.max_playbook_bullets and bullet_count >= self.max_playbook_bullets:
                break
        
        if bullet_count > 0:
            logger.info(f"Loaded {bullet_count} playbook entries from {playbook_path}")
        else:
            logger.info(f"Playbook {playbook_path} is empty or contains no content.")
        
        return playbook_text
    
    def _check_appworld(self):
        """Verify AppWorld package availability"""
        try:
            from appworld import AppWorld
            logger.debug("AppWorld package successfully imported.")
        except ImportError:
            logger.error("AppWorld package not found. Run: uv pip install appworld")
            raise
    
    def load_task(self, task_id: str) -> Dict:
        """Load task specification"""
        task_dir = self.data_dir / "tasks" / task_id
        specs_file = task_dir / "specs.json"

        if not specs_file.exists():
            logger.error(f"Task specification not found: {specs_file}")
            raise FileNotFoundError(f"Task specification not found: {specs_file}")
        
        with open(specs_file, 'r') as f:
            specs = json.load(f)
        
        return {
            'id': task_id,
            'instruction': specs.get('instruction', ''),
            'datetime': specs.get('datetime', ''),
            'supervisor': specs.get('supervisor', {})
        }
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response"""
        code_match = re.search(r'```python\s*\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        logger.warning("No ```python...``` block found in LLM response.")
        return ""

    def _execute_code(self, world, code: str) -> Tuple[str, bool]:
        """Execute code in AppWorld environment with error handling"""
        try:
            output = world.execute(code)
            
            if isinstance(output, str) and output.startswith("Execution failed"):
                return output, False
            if output is None:
                return "None", True
            return str(output), True
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return f"Execution error: {str(e)}", False
        
    def _load_ground_truth(self, task_id: str) -> Dict:
        """Load ground truth data for validation"""
        task_dir = self.data_dir / "tasks" / task_id / "ground_truth"
        ground_truth = {'answer': None, 'private_data': None}
        
        # Load expected answer
        answer_file = task_dir / "answer.json"
        if answer_file.exists():
            try:
                with open(answer_file, 'r') as f:
                    answer_data = json.load(f)
                    ground_truth['answer'] = (
                        answer_data['answer'] if isinstance(answer_data, dict) and 'answer' in answer_data
                        else answer_data
                    )
            except Exception as e:
                logger.warning(f"Could not load ground truth answer for {task_id}: {e}")
        
        # Load private validation data
        private_data_file = task_dir / "private_data.json"
        if private_data_file.exists():
            try:
                with open(private_data_file, 'r') as f:
                    ground_truth['private_data'] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load private data for {task_id}: {e}")
        
        return ground_truth

    def run_episode(self, task_id: str, experiment_name: str = "react_agent", verbose: bool = True) -> Episode:
        """
        Execute complete ReAct episode for a given task.
        Uses AppWorld's built-in evaluation system to determine success based on test results.
        """
        from appworld import AppWorld
        
        try:
            task = self.load_task(task_id)
        except FileNotFoundError as e:
            logger.error(f"Failed to load task {task_id}: {e}")
            return Episode(task_id=task_id, instruction="", error=f"Task file not found: {e}")

        supervisor = task['supervisor']
        main_user = {
            'first_name': supervisor.get('first_name', ''),
            'last_name': supervisor.get('last_name', ''),
            'email': supervisor.get('email', ''),
            'phone_number': supervisor.get('phone_number', '')
        }
        
        episode = Episode(task_id=task_id, instruction=task['instruction'])
        
        logger.info(f"--- Starting Episode: {task_id} ---")
        logger.info(f"Instruction: {task['instruction']}")
        
        conversation_history = ""
        consecutive_no_code_steps = 0
        max_consecutive_no_code = 3  # Allow up to 3 consecutive steps without code (for CoT)
        
        try:
            with AppWorld(task_id=task_id, experiment_name=experiment_name) as world:
                for step_num in range(self.max_steps):
                    logger.info(f"--- Step {step_num + 1}/{self.max_steps} ---")
                    
                    try:
                        # 1. Generate reasoning and action
                        prompt = get_generator_prompt(
                            playbook_json=self.playbook,
                            task=task['instruction'],
                            main_user=main_user,
                            conversation_history=conversation_history
                        )
                        
                        logger.debug("Generating thought and action...")
                        response = self.llm._call_llm(prompt)
                        logger.debug(f"LLM Response: {response[:300]}...")
                        
                        # 2. Extract and validate code
                        code = self._extract_code(response)
                        
                        if not code:
                            consecutive_no_code_steps += 1
                            logger.warning(f"No executable code generated by LLM (attempt {consecutive_no_code_steps}/{max_consecutive_no_code}).")
                            
                            # Allow CoT reasoning without code for a few steps
                            if consecutive_no_code_steps >= max_consecutive_no_code:
                                logger.error(f"No code generated for {max_consecutive_no_code} consecutive steps. Stopping.")
                                episode.error = f"No code generated for {max_consecutive_no_code} consecutive steps"
                                break
                            
                            # Record the thought-only step (CoT)
                            step = Step(
                                step_num=step_num,
                                thought=response,
                                action="",  # No action taken
                                observation="No code to execute. Continue reasoning.",
                                success=False
                            )
                            episode.steps.append(step)
                            
                            # Update conversation with the reasoning
                            conversation_history += (
                                f"\n\n--- Step {step_num + 1} ---\n"
                                f"Reasoning:\n{response}\n\n"
                                f"Note: No executable code was provided. Please provide code to execute.\n"
                            )
                            continue
                        
                        # Reset counter if code was found
                        consecutive_no_code_steps = 0
                        
                        logger.info(f"Action:\n```python\n{code}\n```")
                        
                        # 3. Execute action
                        observation, success = self._execute_code(world, code)
                        
                        logger.info(f"Observation: {observation[:300]}...")
                        logger.info(f"Success: {success}")
                        
                        # 4. Record step
                        step = Step(
                            step_num=step_num,
                            thought=response,
                            action=code,
                            observation=observation,
                            success=success
                        )
                        episode.steps.append(step)
                        
                        # 5. Update conversation context
                        conversation_history += (
                            f"\n\n--- Step {step_num + 1} ---\n"
                            f"Code:\n```python\n{code}\n```\n\n"
                            f"Output:\n{observation}\n"
                        )
                        
                        # 6. Check if agent explicitly called complete_task()
                        if 'complete_task' in code:
                            logger.info("Agent called complete_task() function - will evaluate after this step")
                            break
                    
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Step error: {error_msg}", exc_info=True)
                        episode.error = error_msg
                        break
                
                # Use AppWorld's built-in evaluate() method for comprehensive evaluation
                if verbose:
                    logger.info("\n" + "="*80)
                    logger.info("APPWORLD EVALUATION")
                    logger.info("="*80)
                
                try:
                    evaluation_result = world.evaluate()
                    
                    if verbose:
                        # Display the evaluation result
                        logger.info(str(evaluation_result))
                    
                    # Generate the report (this saves the markdown file)
                    if hasattr(evaluation_result, 'report'):
                        evaluation_report = evaluation_result.report()
                        if verbose:
                            logger.info("\nDetailed Evaluation Report:")
                            logger.info(evaluation_report)
                    
                    # Load and parse the saved report.md file
                    report_path = Path("experiments") / "outputs" / experiment_name / "tasks" / task_id / "evaluation" / "report.md"
                    
                    passed_tests = 0
                    failed_tests = 0
                    total_tests = 0
                    
                    if report_path.exists():
                        try:
                            with open(report_path, 'r', encoding='utf-8') as f:
                                report_content = f.read()
                            
                            if verbose:
                                logger.info(f"\nLoaded evaluation report from: {report_path}")
                            
                            # Parse the markdown report for test statistics
                            # Look for patterns like:
                            # Num Passed Tests : 1
                            # Num Failed Tests : 1
                            # Num Total  Tests : 2
                            
                            passed_match = re.search(r'Num\s+Passed\s+Tests\s*:\s*(\d+)', report_content)
                            failed_match = re.search(r'Num\s+Failed\s+Tests\s*:\s*(\d+)', report_content)
                            total_match = re.search(r'Num\s+Total\s+Tests\s*:\s*(\d+)', report_content)
                            
                            if passed_match:
                                passed_tests = int(passed_match.group(1))
                            if failed_match:
                                failed_tests = int(failed_match.group(1))
                            if total_match:
                                total_tests = int(total_match.group(1))
                            
                            if verbose:
                                logger.info(f"\nParsed Test Results from report.md:")
                                logger.info(f"  Passed Tests: {passed_tests}")
                                logger.info(f"  Failed Tests: {failed_tests}")
                                logger.info(f"  Total Tests: {total_tests}")
                        
                        except Exception as parse_error:
                            logger.warning(f"Failed to parse report.md: {parse_error}")
                            # Fall back to trying to extract from evaluation_result object
                            if hasattr(evaluation_result, 'num_passed'):
                                passed_tests = evaluation_result.num_passed
                            if hasattr(evaluation_result, 'num_failed'):
                                failed_tests = evaluation_result.num_failed
                            if hasattr(evaluation_result, 'num_tests'):
                                total_tests = evaluation_result.num_tests
                    else:
                        logger.warning(f"Report file not found at: {report_path}")
                        # Try to extract from evaluation_result object directly
                        if hasattr(evaluation_result, 'num_passed'):
                            passed_tests = evaluation_result.num_passed
                        if hasattr(evaluation_result, 'num_failed'):
                            failed_tests = evaluation_result.num_failed
                        if hasattr(evaluation_result, 'num_tests'):
                            total_tests = evaluation_result.num_tests
                    
                    # Determine success based on test results
                    # Success = all tests passed AND no tests failed
                    if total_tests > 0:
                        if not (passed_tests == total_tests and failed_tests == 0) and total_tests > 2:
                            episode.success = (passed_tests == total_tests-1 and failed_tests == 1)
                        else:
                            episode.success = (passed_tests == total_tests and failed_tests == 0)
                    else:
                        # If we can't extract test counts, fall back to task_completed
                        logger.warning("Could not extract test counts from report, falling back to task_completed")
                        episode.success = world.task_completed
                    
                    # Store evaluation metadata
                    episode.metadata = {
                        'passed_tests': passed_tests,
                        'failed_tests': failed_tests,
                        'total_tests': total_tests,
                        'task_completed': world.task_completed,
                        'evaluation_result': str(evaluation_result),
                        'report_path': str(report_path) if report_path.exists() else None
                    }
                    
                    if verbose:
                        logger.info(f"\n{'='*60}")
                        logger.info(f"FINAL TEST RESULTS")
                        logger.info(f"{'='*60}")
                        logger.info(f"  Passed: {passed_tests}/{total_tests}")
                        logger.info(f"  Failed: {failed_tests}/{total_tests}")
                        logger.info(f"  Success: {'✓ YES' if episode.success else '✗ NO'}")
                        logger.info(f"{'='*60}")
                    
                except Exception as eval_error:
                    logger.error(f"Evaluation error: {eval_error}", exc_info=True)
                    # Fallback to task_completed property
                    episode.success = world.task_completed
                    episode.metadata = {
                        'task_completed': world.task_completed,
                        'evaluation_error': str(eval_error)
                    }

                if len(episode.steps) == self.max_steps and not episode.success:
                    logger.warning(f"Max steps ({self.max_steps}) reached without completing task.")
                    if not episode.error:
                        episode.error = "Max steps reached"

        except Exception as e:
            logger.error(f"AppWorld Environment error: {e}", exc_info=True)
            episode.error = str(e)
        
        # Final summary
        logger.info(f"\n{'='*80}")
        logger.info(f"EPISODE SUMMARY: {task_id}")
        logger.info(f"{'='*80}")
        logger.info(f"Status: {'✓ SUCCESS' if episode.success else '✗ FAILED'}")
        logger.info(f"Steps Taken: {len(episode.steps)}/{self.max_steps}")
        
        if hasattr(episode, 'metadata') and episode.metadata:
            logger.info(f"Test Results: {episode.metadata.get('passed_tests', 0)}/{episode.metadata.get('total_tests', 0)} passed, "
                       f"{episode.metadata.get('failed_tests', 0)} failed")
        
        if episode.error:
            logger.warning(f"Error: {episode.error}")
        logger.info(f"{'='*80}\n")
        
        return episode

