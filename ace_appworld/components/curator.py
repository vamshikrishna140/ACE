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
from ace_appworld.components.llm import LLM

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
        self.dedup_threshold = config.CURATOR_DEDUP_THRESHOLD
        self.model_provider = config.CURATOR_PROVIDER
        self.llm = LLM(model_name=self.model_name, model_provider=self.model_provider)

        # Initialize embedding model for deduplication
        self._init_embeddings()
        
        logger.info(f"Curator initialized: Provider={self.model_provider}, Model={self.model_name}")
    
    
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

            response = self.llm._call_llm(prompt)
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


