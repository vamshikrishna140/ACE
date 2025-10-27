
import os
import json
import logging
from typing import Dict, List
from ace_appworld.components.curator import Curator, CurationOperation
from ace_appworld import config
logger = logging.getLogger(__name__)

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

