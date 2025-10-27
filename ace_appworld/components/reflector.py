import json
import os
import re
import logging
from typing import Dict, List, Optional, Tuple
from ace_appworld import config
from ace_appworld.components.models import ReflectionResult
from ace_appworld.components.prompts import get_reflection_prompt
from ace_appworld.components.llm import LLM

# Setup logger
logger = logging.getLogger(__name__)

class Reflector:
    """
    ACE Reflector Component
    Analyzes trajectories to identify errors, root causes, and actionable insights.
    """
    
    def __init__(self):
        
        self.model_name = config.REFLECTOR_MODEL
        self.max_refinement_rounds = config.REFLECTOR_MAX_REFINEMENT
        self.model_provider = config.REFLECTOR_PROVIDER
        self.llm = LLM(model_name=self.model_name, model_provider=self.model_provider)
        logger.info(f"Reflector initialized: Provider={self.model_provider}, Model={self.model_name}")
    
    
    
    def _build_reflection_prompt(self,
                                 task_instruction: str,
                                 trajectory: List[Dict],
                                 final_answer: Optional[str],
                                 ground_truth: Optional[Dict],
                                 execution_feedback: str,
                                 playbook_bullets: Optional[List[Dict]] = None) -> str:
        """
        Build reflection prompt for trajectory analysis (using paper's exact prompt)
        """
        prompt = get_reflection_prompt(
            task_instruction=task_instruction,
            trajectory=trajectory,
            final_answer=final_answer,
            ground_truth=ground_truth,
            execution_feedback=execution_feedback,
            playbook_bullets=playbook_bullets
        )
        
        return prompt
    
    def _parse_json_from_response(self, response: str) -> Dict:
        """Robustly parse JSON from LLM response, handling markdown."""
        try:
            # Try to find JSON in markdown
            match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # Try to find raw JSON object
                match = re.search(r'\{\s*".*?\}\s*$', response, re.DOTALL)
                if match:
                    json_str = match.group(0)
                else:
                    json_str = response # Assume it's raw JSON

            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            logger.error(f"Reflection JSON parsing error: {e}. Raw response: {response[:500]}...")
            raise ValueError(f"Failed to parse JSON from Reflector response. Content: {response}")

    def reflect(self,
                task_instruction: str,
                trajectory: List[Dict],
                final_answer: Optional[str] = None,
                ground_truth: Optional[Dict] = None,
                execution_feedback: str = "",
                playbook_bullets: Optional[List[Dict]] = None) -> ReflectionResult:
        """
        Perform reflection on agent trajectory
        """
        logger.info(f"Reflecting on trajectory... (Instruction: {task_instruction[:50]}...)")
        
        # --- Run reflection (potentially with refinement) ---
        reflections = []
        augmented_feedback = execution_feedback
        
        for round_num in range(self.max_refinement_rounds):
            logger.debug(f"Reflection round {round_num + 1}/{self.max_refinement_rounds}")
            
            prompt = self._build_reflection_prompt(
                task_instruction=task_instruction,
                trajectory=trajectory,
                final_answer=final_answer,
                ground_truth=ground_truth,
                execution_feedback=augmented_feedback,
                playbook_bullets=playbook_bullets
            )
            
            response = self.llm._call_llm(prompt)
            reflection_data = self._parse_json_from_response(response)
            
            current_reflection = ReflectionResult(
                reasoning=reflection_data.get("reasoning", ""),
                error_identification=reflection_data.get("error_identification", ""),
                root_cause_analysis=reflection_data.get("root_cause_analysis", ""),
                correct_approach=reflection_data.get("correct_approach", ""),
                key_insight=reflection_data.get("key_insight", ""),
                bullet_tags=reflection_data.get("bullet_tags")
            )
            reflections.append(current_reflection)
            
            # Add this reflection's insight to the feedback for the next round
            augmented_feedback += f"\n\n**Previous Reflection (Round {round_num + 1}):**\n"
            augmented_feedback += f"Insight: {current_reflection.key_insight}\n"
            
            # Simple convergence check (if more than one round)
            if len(reflections) > 1:
                prev_insight = reflections[-2].key_insight.lower().strip()
                curr_insight = current_reflection.key_insight.lower().strip()
                if prev_insight == curr_insight:
                    logger.info(f"Reflection converged after {round_num + 1} rounds.")
                    break
        
        # --- Select the best (last) reflection ---
        best_reflection = reflections[-1]
        
        logger.info(f"Reflection complete. Key Insight: {best_reflection.key_insight[:100]}...")
        return best_reflection

