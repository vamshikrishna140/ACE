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
from ace_appworld import config

# Setup logger
logger = logging.getLogger(__name__)

class ReActAgent:
    """
    ReAct Agent (Generator) for AppWorld Environment
    Implements the Reason-Act-Observe loop with enhanced validation.
    """
    
    def __init__(self, 
             playbook_path: str = config.PLAYBOOK_PATH,
             use_playbook: bool = True,
             max_playbook_bullets: Optional[int] = None,
             enable_validation: bool = True):
        
        self.data_dir = config.APPWORLD_DATA_DIR
        self.model_provider = config.GENERATOR_PROVIDER
        self.model_name = config.GENERATOR_MODEL
        self.max_steps = config.MAX_EPISODE_STEPS
        self.temperature = config.TEMPERATURE
        self.max_tokens = config.MAX_TOKENS
        
        self.use_playbook = use_playbook
        self.max_playbook_bullets = max_playbook_bullets
        self.enable_validation = enable_validation
        
        # Initialize the appropriate client
        if self.model_provider == "gemini":
            self.api_key = config.GEMINI_API_KEY
            self._init_gemini_client()
        elif self.model_provider == "ollama":
            self.api_key = config.OLLAMA_API_KEY
            self._init_ollama_client()
        elif self.model_provider == "openrouter":
            self.api_key = config.OPENROUTER_API_KEY
            self._init_openrouter_client()
        else:
            raise ValueError(f"Unknown model_provider in config: {self.model_provider}")
            
        # Load domain knowledge
        self.playbook = self._load_playbook(playbook_path) if use_playbook else ""
        
        # Verify dependencies
        self._check_appworld()
        
        logger.info(f"ReAct Agent initialized: Provider={self.model_provider}, Model={self.model_name}")
    
    def update_playbook(self, playbook_path: str):
        """Public method to reload the playbook from disk."""
        logger.info("Agent is reloading playbook...")
        self.playbook = self._load_playbook(playbook_path)

    def _init_openrouter_client(self):
        if not self.api_key:
            raise ValueError("OpenRouter API key not set in .env or config.")
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        logger.debug("OpenRouter client configured")

    def _init_gemini_client(self):
        try:
            from google import genai
            from google.genai import types
            if not self.api_key:
                raise ValueError("Gemini API key not set in .env or config.")
            os.environ["GOOGLE_API_KEY"] = self.api_key
            self.gemini_client = genai.Client()
            self.gemini_types = types
            logger.debug("Gemini client initialized")
        except ImportError:
            logger.error("google-genai not installed. Run: uv pip install google-genai")
            raise
    
    def _init_ollama_client(self):
        self.api_key = self.api_key or config.OLLAMA_API_KEY
        if not self.api_key:
            raise ValueError("Ollama Cloud API key not set in .env or config.")
        self.ollama_cloud_url = "https://ollama.com"
        logger.debug("Ollama client configured")
        
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

    def _extract_final_answer(self, code: str, observation: str) -> Optional[str]:
        """Extract final answer from code execution"""
        # Pattern 1: complete_task(answer="...")
        match = re.search(r'complete_task\s*\(\s*answer\s*=\s*["\'](.+?)["\']\s*\)', code)
        if match:
            return match.group(1)
        
        # Pattern 2: complete_task(answer=variable)
        match = re.search(r'complete_task\s*\(\s*answer\s*=\s*(\w+)\s*\)', code)
        if match:
            var_name = match.group(1)
            var_match = re.search(rf'{var_name}\s*=\s*["\'](.+?)["\']', code)
            if var_match:
                return var_match.group(1)
        
        # Pattern 3: Return value in observation
        if 'complete_task' in code and observation and observation != "None":
            obs_clean = observation.strip()
            if obs_clean and not obs_clean.startswith("Execution failed"):
                return obs_clean
        
        # Pattern 4: Variable assignment followed by return
        match = re.search(r'answer\s*=\s*["\'](.+?)["\']', code)
        if match and 'complete_task' in code:
            return match.group(1)
        
        return None
    
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
    
    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM"""
        try:
            if self.model_provider == "gemini":
                return self._call_gemini(prompt)
            elif self.model_provider == "openrouter":
                return self._call_openrouter(prompt)
            elif self.model_provider == "ollama":
                return self._call_ollama_cloud(prompt)
            else:
                raise ValueError(f"Provider {self.model_provider} not supported")
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            raise
        
    def _call_openrouter(self, prompt: str) -> str:
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": {"sort": "price"}
        }
        response = requests.post(self.openrouter_url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def _call_gemini(self, prompt: str) -> str:
        config_gemini = self.gemini_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens
        )
        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config_gemini
        )
        return response.text
    
    def _call_ollama_cloud(self, prompt: str) -> str:
        from ollama import Client
        client = Client(host=self.ollama_cloud_url, headers={'Authorization': f'Bearer {self.api_key}'})
        messages = [{"role": "user", "content": prompt}]
        response = client.chat(
            model=self.model_name,
            messages=messages,
            options={"temperature": self.temperature, "num_predict": self.max_tokens, "timeout": 180}
        )
        return response['message']['content']

    def _validate_trajectory(self, episode: Episode, ground_truth: Dict, verbose: bool = False) -> bool:
        """
        Validate episode trajectory against ground truth
        Uses LLM-based semantic validation if enabled, otherwise uses rule-based validation
        """
        if not self.enable_validation:
            return self._rule_based_validation(episode, ground_truth)
        
        try:
            validation_prompt = self._build_validation_prompt(episode, ground_truth)
            
            if verbose:
                logger.info("🔍 Validating trajectory with LLM...")
            
            response = self._call_llm(validation_prompt)
            is_valid = self._parse_validation_response(response)
            
            if verbose:
                logger.info(f"Validation Result: {'✓ Valid' if is_valid else '✗ Invalid'}")
                if not is_valid:
                    logger.info(f"Validation Reason: {response[:200]}...")
            
            return is_valid
            
        except Exception as e:
            logger.warning(f"⚠ Validation error: {e}. Falling back to rule-based validation")
            return self._rule_based_validation(episode, ground_truth)
    
    def _build_validation_prompt(self, episode: Episode, ground_truth: Dict) -> str:
        """Build validation prompt for LLM-based trajectory validation"""
        prompt = f"""You are a validation agent for task completion verification.

**Task Instruction:**
{episode.instruction}

**Agent Execution Trajectory:**
"""
        
        for i, step in enumerate(episode.steps, 1):
            prompt += f"""
--- Step {i} ---
Reasoning: {step.thought}

Action:
```python
{step.action}
```

Observation: {step.observation}
Success: {step.success}
"""
        
        prompt += f"""
**Final Answer:** {episode.final_answer if episode.final_answer else "No explicit answer provided"}

**Ground Truth:**
"""
        
        if ground_truth['answer'] is not None:
            prompt += f"Expected Answer: {json.dumps(ground_truth['answer'], indent=2)}\n"
        else:
            prompt += "Expected Answer: Not explicitly specified\n"
        
        if ground_truth['private_data']:
            prompt += f"\nValidation Data: {json.dumps(ground_truth['private_data'], indent=2)}\n"
        
        prompt += """
**Validation Criteria:**
1. Analyze each step in the trajectory to understand the agent's reasoning and actions
2. Check if the sequence of actions logically accomplishes the task instruction
3. Compare the final answer (if provided) with the ground truth
4. If no explicit answer, verify that the actions successfully completed the task
5. Consider semantic equivalence (e.g., "yes"/"true", date format variations, equivalent phrasings)
6. Review observations to confirm correct data was retrieved or modified
7. Check if the final action indicates task completion

**Important Notes:**
- Exact string matching is NOT required; focus on semantic correctness
- The task may be completed through actions without an explicit answer
- Consider the overall trajectory, not just the final answer
- Verify that side effects (updates, deletions, creations) were properly executed

**Response Format:**
Respond with ONLY one of:
- SUCCESS: [concise reason why the task was completed correctly]
- FAILURE: [concise reason why the task was not completed or completed incorrectly]

Be decisive and clear in your judgment.
"""
        
        return prompt
    
    def _parse_validation_response(self, response: str) -> bool:
        """Parse validation response into boolean success indicator"""
        response_lower = response.lower().strip()
        
        # Check for explicit success/failure markers
        if response_lower.startswith("success"):
            return True
        elif response_lower.startswith("failure"):
            return False
        
        # Fallback: count positive vs negative indicators
        success_words = ["success", "correct", "accomplished", "completed", 
                        "matches", "valid", "properly", "successfully"]
        failure_words = ["failure", "incorrect", "failed", "wrong", "mismatch", 
                        "invalid", "missing", "incomplete"]
        
        success_count = sum(1 for word in success_words if word in response_lower)
        failure_count = sum(1 for word in failure_words if word in response_lower)
        
        # Require clear positive signal
        return success_count > failure_count and success_count > 0
    
    def _rule_based_validation(self, episode: Episode, ground_truth: Dict) -> bool:
        """
        Rule-based validation using ground truth comparison
        Fallback when LLM validation is unavailable
        """
        # Check for execution errors
        if episode.error or not episode.steps:
            logger.warning(f"Rule-based validation failed: Episode has error or no steps")
            return False
        
        # Check final answer against ground truth
        if episode.final_answer and ground_truth['answer'] is not None:
            is_equivalent = self._check_semantic_equivalence(
                episode.final_answer, 
                ground_truth['answer']
            )
            if is_equivalent:
                logger.info(f"Rule-based validation: Answer matches ground truth")
            else:
                logger.warning(f"Rule-based validation: Answer mismatch. Got: '{episode.final_answer}', Expected: '{ground_truth['answer']}'")
            return is_equivalent
        
        # Check if task completion was indicated
        for step in reversed(episode.steps):
            if 'complete_task' in step.action and step.success:
                logger.info("Rule-based validation: complete_task called successfully")
                return True
        
        # Check private data for implicit validation
        if ground_truth['private_data']:
            is_valid = self._validate_against_private_data(episode, ground_truth['private_data'])
            if is_valid:
                logger.info("Rule-based validation: Validated against private data")
            return is_valid
        
        logger.warning("Rule-based validation failed: No validation criteria met")
        return False
    
    def _check_semantic_equivalence(self, answer: str, expected) -> bool:
        """Check semantic equivalence between answer and expected value"""
        answer_str = str(answer).lower().strip()
        expected_str = str(expected).lower().strip()
        
        # Exact match
        if answer_str == expected_str:
            return True
        
        # Boolean equivalence
        true_values = {'yes', 'true', '1', 'correct', 'success'}
        false_values = {'no', 'false', '0', 'incorrect', 'failure'}
        
        if answer_str in true_values and expected_str in true_values:
            return True
        if answer_str in false_values and expected_str in false_values:
            return True
        
        # Normalize and compare (remove punctuation, extra whitespace)
        answer_normalized = re.sub(r'[^\w\s]', '', answer_str).strip()
        expected_normalized = re.sub(r'[^\w\s]', '', expected_str).strip()
        
        return answer_normalized == expected_normalized
    
    def _validate_against_private_data(self, episode: Episode, private_data: Dict) -> bool:
        """Validate episode trajectory against private validation data"""
        # Check if any observations contain expected private data values
        private_str = json.dumps(private_data).lower()
        
        for step in episode.steps:
            if step.observation and step.observation.lower() in private_str:
                return True
        
        return False

    def run_episode(self, task_id: str, experiment_name: str = "react_agent", verbose: bool = True) -> Episode:
        """
        Execute complete ReAct episode for a given task
        """
        from appworld import AppWorld
        
        try:
            task = self.load_task(task_id)
            ground_truth = self._load_ground_truth(task_id)
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
                        response = self._call_llm(prompt)
                        logger.debug(f"LLM Response: {response[:300]}...")
                        
                        # 2. Extract and validate code
                        code = self._extract_code(response)
                        
                        if not code:
                            logger.warning("No executable code generated by LLM.")
                            episode.error = "No code generated"
                            break
                        
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
                        
                        # 6. Check for task completion
                        if 'complete_task' in code:
                            logger.info("Task completion detected.")
                            final_answer = self._extract_final_answer(code, observation)
                            if final_answer:
                                episode.final_answer = final_answer
                                logger.info(f"Final Answer: {final_answer}")
                            break
                    
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Step error: {error_msg}", exc_info=True)
                        episode.error = error_msg
                        break
                
                if len(episode.steps) == self.max_steps:
                    logger.warning(f"Max steps ({self.max_steps}) reached.")
                    episode.error = "Max steps reached"

        except Exception as e:
            logger.error(f"AppWorld Environment error: {e}", exc_info=True)
            episode.error = str(e)
        
        # 7. Validate trajectory with enhanced validation
        if not episode.error and episode.steps:
            episode.success = self._validate_trajectory(episode, ground_truth, verbose=verbose)
        
        logger.info(f"--- Episode Summary: {task_id} ---")
        logger.info(f"Status: {'✓ Success' if episode.success else '✗ Failed'}")
        logger.info(f"Steps: {len(episode.steps)}")
        logger.info(f"Final Answer: {episode.final_answer if episode.final_answer else 'None'}")
        if episode.error:
            logger.warning(f"Error: {episode.error}")
        logger.info("--- End of Episode ---")
        
        return episode