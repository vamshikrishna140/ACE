import json
import os
import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from ace_appworld import config
from ace_appworld.components.models import CurationOperation, CurationResult, ReflectionResult
from ace_appworld.components.prompts import get_curation_prompt

# Setup logger
logger = logging.getLogger(__name__)

"""
This file contains the Curator (which generates updates)
and the PlaybookManager (which applies them).
(Refactored from your `curator.py`)
"""


class Curator:
    """
    ACE Curator Component
    Synthesizes reflections into structured playbook updates (ADD/UPDATE/DELETE).
    """
    
    def __init__(self):
        self.model_provider = config.CURATOR_PROVIDER
        self.model_name = config.CURATOR_MODEL
        self.temperature = config.TEMPERATURE
        self.max_tokens = config.MAX_TOKENS
        self.dedup_threshold = config.CURATOR_DEDUP_THRESHOLD
        
        # Initialize client
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
        
        # Initialize embedding model for deduplication
        self._init_embeddings()
        
        logger.info(f"Curator initialized: Provider={self.model_provider}, Model={self.model_name}")
    
    def _init_openrouter(self):
        if not self.api_key:
            raise ValueError("OpenRouter API key not set in .env or config.")
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        logger.debug("Curator OpenRouter client configured")
    
    def _init_gemini(self):
        try:
            from google import genai
            from google.genai import types
            if not self.api_key:
                raise ValueError("Gemini API key not set in .env or config.")
            os.environ["GOOGLE_API_KEY"] = self.api_key
            self.gemini_client = genai.Client()
            self.gemini_types = types
            logger.debug("Curator Gemini client initialized")
        except ImportError:
            logger.error("google-genai not installed. Run: uv pip install google-genai")
            raise
    
    def _init_ollama(self):
        if not self.api_key:
            raise ValueError("Ollama Cloud API key not set in .env or config.")
        self.ollama_cloud_url = "https://ollama.com"
        logger.debug("Curator Ollama client configured")
    
    def _init_embeddings(self):
        """Initialize embedding model for deduplication"""
        try:
            from sentence_transformers import SentenceTransformer
            # Using a small, fast model
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence embeddings (all-MiniLM-L6-v2) enabled for deduplication.")
        except ImportError:
            logger.warning("sentence-transformers not installed, deduplication will be disabled. Run: uv pip install sentence-transformers")
            self.embedder = None
    
    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM"""
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
            logger.error(f"Curator LLM call failed: {e}", exc_info=True)
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

    def _build_curation_prompt(self,
                              task_context: str,
                              current_playbook: Dict,
                              reflection: ReflectionResult,
                              generated_code: Optional[str] = None) -> str:
        """Build curation prompt for playbook updates"""
        
        prompt = get_curation_prompt(
            task_context=task_context,
            current_playbook=current_playbook,
            reflection=reflection
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
            logger.error(f"Curation JSON parsing error: {e}. Raw response: {response[:500]}...")
            raise ValueError(f"Failed to parse JSON from Curator response. Content: {response}")

    def curate(self,
              task_context: str,
              current_playbook: Dict,
              reflection: ReflectionResult) -> CurationResult:
        """
        Generate curation operations from a single reflection
        """
        logger.info(f"Curating playbook from reflection (Insight: {reflection.key_insight[:50]}...)")
        
        try:
            prompt = self._build_curation_prompt(
                task_context=task_context,
                current_playbook=current_playbook,
                reflection=reflection
            )
            
            response = self._call_llm(prompt)
            curation_data = self._parse_json_from_response(response)
            
            # Build operations
            operations = []
            for op_data in curation_data.get('operations', []):
                if not op_data.get('section') or not op_data.get('content'):
                    logger.warning(f"Skipping malformed curation op: {op_data}")
                    continue
                
                # Only allow ADD operations
                if op_data.get('type') == "ADD":
                    operations.append(CurationOperation(
                        type="ADD",
                        section=op_data.get('section'),
                        content=op_data.get('content'),
                        reason=curation_data.get('reasoning', '')
                    ))
            
            result = CurationResult(
                reasoning=curation_data.get('reasoning', ''),
                operations=operations
            )
            
            logger.info(f"Curator generated {len(operations)} new ADD operations.")
            return result
        
        except Exception as e:
            logger.error(f"Curation failed: {e}", exc_info=True)
            return CurationResult(reasoning=f"Curation failed: {e}", operations=[])
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        if not self.embedder:
            logger.debug("Using fallback word overlap for similarity.")
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1 & words2) / max(len(words1), len(words2))
        
        try:
            embeddings = self.embedder.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}. Disabling embedder.")
            self.embedder = None # Disable for future calls
            return self.compute_similarity(text1, text2) # Recurse
    
    def deduplicate_operations(self, 
                              operations: List[CurationOperation],
                              existing_playbook: Dict) -> List[CurationOperation]:
        """
        Deduplicate operations against existing playbook
        """
        if not operations:
            return operations
        
        logger.info(f"Deduplicating {len(operations)} new operations...")
        deduplicated_ops = []
        
        for op in operations:
            if op.type != "ADD" or not op.content:
                deduplicated_ops.append(op)
                continue
            
            section_bullets = existing_playbook.get(op.section, [])
            is_duplicate = False
            for existing_bullet in section_bullets:
                existing_content = existing_bullet.get('content', '')
                if not existing_content:
                    continue
                
                similarity = self.compute_similarity(op.content, existing_content)
                
                if similarity >= self.dedup_threshold:
                    logger.warning(f"Duplicate insight (Sim={similarity:.2f}). Skipping ADD.")
                    logger.debug(f"  New: {op.content[:80]}...")
                    logger.debug(f"  Existing: {existing_content[:80]}...")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated_ops.append(op)
        
        removed = len(operations) - len(deduplicated_ops)
        logger.info(f"Deduplication complete. Removed {removed} operations.")
        return deduplicated_ops


class PlaybookManager:
    """
    High-level playbook management (Executor)
    Applies operations from the Curator to the `playbook.json` file.
    """
    
    def __init__(self, curator: Curator):
        self.curator = curator
    
    def load_playbook(self, path: str) -> Dict:
        """Load playbook from JSON file"""
        if not os.path.exists(path):
            logger.warning(f"Playbook file not found: {path}. Returning empty playbook.")
            return {}
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {path}. Returning empty playbook.")
            return {}
        except Exception as e:
            logger.error(f"Failed to load playbook {path}: {e}. Returning empty playbook.")
            return {}
    
    def save_playbook(self, playbook: Dict, path: str):
        """Save playbook to JSON file"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(playbook, f, indent=2, ensure_ascii=False)
            logger.info(f"Playbook saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save playbook to {path}: {e}")
    
    def get_playbook_stats(self, playbook: Dict) -> Dict:
        """Get statistics about playbook"""
        stats = {'total_bullets': 0, 'sections': {}, 'helpful_bullets': 0, 'harmful_bullets': 0}
        for section, bullets in playbook.items():
            if not isinstance(bullets, list): continue
            count = len(bullets)
            stats['total_bullets'] += count
            stats['sections'][section] = count
            for bullet in bullets:
                if bullet.get('helpful', 0) > 0: stats['helpful_bullets'] += 1
                if bullet.get('harmful', 0) > 0: stats['harmful_bullets'] += 1
        return stats
    
    def apply_operations(self,
                        playbook: Dict,
                        operations: List[CurationOperation]) -> Dict:
        """
        Apply curation operations to playbook
        """
        logger.info(f"Applying {len(operations)} operations to playbook...")
        updated_playbook = json.loads(json.dumps(playbook))  # Deep copy
        
        # Find max ID to avoid collisions
        max_id = 0
        for section, bullets in updated_playbook.items():
            if not isinstance(bullets, list): continue
            for bullet in bullets:
                bullet_id = bullet.get('id', '')
                if bullet_id and '-' in bullet_id:
                    try:
                        num = int(bullet_id.split('-')[-1])
                        max_id = max(max_id, num)
                    except ValueError:
                        continue
        
        for op in operations:
            if op.type == "ADD":
                max_id += 1
                section_prefix = op.section[:3] if op.section else "gen"
                new_bullet = {
                    'id': f"{section_prefix}-{max_id:05d}",
                    'content': op.content,
                    'helpful': 0,
                    'harmful': 0
                }
                
                if op.section not in updated_playbook:
                    updated_playbook[op.section] = []
                
                updated_playbook[op.section].append(new_bullet)
                logger.info(f"ADD [{new_bullet['id']}] to {op.section}: {op.content[:80]}...")
            
            # Add UPDATE/DELETE logic here if Curation prompt is expanded
        
        return updated_playbook
    
    def update_bullet_tags(self,
                          playbook: Dict,
                          bullet_tags: List[Dict]) -> Dict:
        """
        Update helpful/harmful counters based on reflection tags
        """
        if not bullet_tags:
            return playbook
        
        logger.info(f"Updating {len(bullet_tags)} bullet tags...")
        updated_playbook = json.loads(json.dumps(playbook))  # Deep copy
        tag_counts = {'helpful': 0, 'harmful': 0, 'neutral': 0}
        
        for tag_data in bullet_tags:
            bullet_id = tag_data.get('id')
            tag = tag_data.get('tag', 'neutral').lower()
            if not bullet_id or tag not in tag_counts:
                continue

            tag_counts[tag] += 1
            
            found = False
            for section, bullets in updated_playbook.items():
                if not isinstance(bullets, list): continue
                for bullet in bullets:
                    if bullet.get('id') == bullet_id:
                        if tag == 'helpful':
                            bullet['helpful'] = bullet.get('helpful', 0) + 1
                        elif tag == 'harmful':
                            bullet['harmful'] = bullet.get('harmful', 0) + 1
                        found = True
                        break
                if found:
                    break
        
        logger.info(f"Tags updated: +{tag_counts['helpful']} helpful, +{tag_counts['harmful']} harmful")
        return updated_playbook
    
    def prune_harmful_bullets(self, playbook: Dict) -> Dict:
        """
        Remove bullets that are consistently harmful based on config thresholds
        """
        harmful_threshold = config.CURATOR_PRUNE_HARMFUL_THRESHOLD
        helpful_ratio = config.CURATOR_PRUNE_HELPFUL_RATIO
        
        logger.info(f"Pruning harmful bullets (Threshold: harm > {harmful_threshold} & ratio < {helpful_ratio})...")
        updated_playbook = json.loads(json.dumps(playbook))  # Deep copy
        removed_count = 0
        
        for section, bullets in list(updated_playbook.items()):
            if not isinstance(bullets, list): continue
            filtered_bullets = []
            for bullet in bullets:
                helpful = bullet.get('helpful', 0)
                harmful = bullet.get('harmful', 0)
                total = helpful + harmful
                
                if harmful >= harmful_threshold and total > 0:
                    ratio = helpful / total
                    if ratio < helpful_ratio:
                        logger.warning(f"Pruning bullet [{bullet['id']}] (H/F={helpful}/{harmful}, Ratio={ratio:.2f})")
                        removed_count += 1
                        continue
                
                filtered_bullets.append(bullet)
            updated_playbook[section] = filtered_bullets
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} harmful bullet(s)")
        else:
            logger.debug("No harmful bullets to prune")
        
        return updated_playbook

