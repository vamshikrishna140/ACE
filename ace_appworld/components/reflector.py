import json
import os
import re
import logging
from typing import Dict, List, Optional, Tuple
from ace_appworld import config
from ace_appworld.components.models import ReflectionResult
from ace_appworld.components.prompts import get_reflection_prompt

# Setup logger
logger = logging.getLogger(__name__)

"""
This file contains the Reflector agent (Critic).
(Refactored from your `Reflector.py`)
"""

class Reflector:
    """
    ACE Reflector Component
    Analyzes trajectories to identify errors, root causes, and actionable insights.
    """
    
    def __init__(self):
        
        self.model_provider = config.REFLECTOR_PROVIDER
        self.model_name = config.REFLECTOR_MODEL
        self.temperature = config.TEMPERATURE
        self.max_tokens = config.MAX_TOKENS
        self.max_refinement_rounds = config.REFLECTOR_MAX_REFINEMENT
        
        # Initialize client based on provider
        if self.model_provider == "openrouter":
            self.api_key = config.OPENROUTER_API_KEY
            self._init_openrouter()
        elif self.model_provider == "gemini":
            self.api_key = config.GEMINI_API_KEY
            self._init_gemini()
        elif self.model_provider == "ollama":
            self.api_key = config.OLLAMA_API_KEY
            self._init_ollama()
        else:
            raise ValueError(f"Unknown provider in config: {self.model_provider}")
        
        logger.info(f"Reflector initialized: Provider={self.model_provider}, Model={self.model_name}")
    
    def _init_openrouter(self):
        if not self.api_key:
            raise ValueError("OpenRouter API key not set in .env or config.")
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        logger.debug("Reflector OpenRouter client configured")
    
    def _init_gemini(self):
        try:
            from google import genai
            from google.genai import types
            if not self.api_key:
                raise ValueError("Gemini API key not set in .env or config.")
            os.environ["GOOGLE_API_KEY"] = self.api_key
            self.gemini_client = genai.Client()
            self.gemini_types = types
            logger.debug("Reflector Gemini client initialized")
        except ImportError:
            logger.error("google-genai not installed. Run: uv pip install google-genai")
            raise
    
    def _init_ollama(self):
        if not self.api_key:
            raise ValueError("Ollama Cloud API key not set in .env or config.")
        self.ollama_cloud_url = "https://ollama.com"
        logger.debug("Reflector Ollama client configured")
    
    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM for reflection"""
        try:
            if self.model_provider == "openrouter":
                return self._call_openrouter(prompt)
            elif self.model_provider == "gemini":
                return self._call_gemini(prompt)
            elif self.model_provider == "ollama":
                return self._call_ollama(prompt)
            else:
                raise ValueError(f"Provider {self.model_provider} not supported")
        except Exception as e:
            logger.error(f"Reflector LLM call failed: {e}", exc_info=True)
            raise

    def _call_openrouter(self, prompt: str) -> str:
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens, 
            "provider": {
                "sort": "price"
            }
        }
        response = requests.post(self.openrouter_url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
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

    def _call_ollama(self, prompt: str) -> str:
        from ollama import Client
        client = Client(host=self.ollama_cloud_url, headers={'Authorization': f'Bearer {self.api_key}'})
        response = client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature, "num_predict": self.max_tokens, "timeout": 180}
        )
        return response['message']['content']
    
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
            
            response = self._call_llm(prompt)
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

