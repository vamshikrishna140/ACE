from typing import Optional, Tuple, Dict
import logging
from ace_appworld.components.thompson_sampling import ThompsonSamplingPolicy


class OnlineRewardTracker:
    """
    Enhanced reward tracker using Thompson Sampling for decisions.
    
    Replaces the old heuristic-based approach with principled Bayesian learning.
    """
    
    def __init__(self, thompson_policy: Optional[ThompsonSamplingPolicy] = None):
        """
        Args:
            thompson_policy: Pre-configured Thompson Sampling policy.
                           If None, creates default policy.
        """
        self.policy = thompson_policy or ThompsonSamplingPolicy(
            attempt_rewards={1: 1.0, 2: 0.6, 3: 0.3},
            cost_per_attempt=0.1,
            use_quality_weighted_attribution=True
        )
        
        self.attempt_history = []
        self.cumulative_reward = 0.0
    
    def calculate_attempt_reward(self, episode, attempt_num: int) -> float:
        """
        Calculate reward for this attempt.
        
        Uses the reward structure from Thompson policy.
        """
        if episode.success:
            base_reward = self.policy.attempt_rewards.get(attempt_num, 0.0)
            
            # Optional: Add efficiency bonus
            if hasattr(episode, 'steps') and episode.steps:
                from ace_appworld import config
                efficiency = (config.MAX_EPISODE_STEPS - len(episode.steps)) / config.MAX_EPISODE_STEPS
                efficiency_bonus = efficiency * 0.2  # Small bonus
                return base_reward + efficiency_bonus
            
            return base_reward
        else:
            # Failure penalty
            return -self.policy.cost_per_attempt * attempt_num
    
    def calculate_reflection_quality(self, reflection, previous_episode) -> float:
        """
        Estimate reflection quality (0-1) based on content analysis.
        
        This is your existing heuristic - kept as-is.
        """
        score = 0.0
        
        # Length heuristics
        if len(reflection.error_identification) > 50:
            score += 0.3
        
        if len(reflection.correct_approach) > 100:
            score += 0.3
        
        if len(reflection.key_insight) > 50:
            score += 0.4
        
        # Penalize vague reflections
        vague_phrases = ['should be careful', 'need to check', 'might be wrong']
        if any(phrase in reflection.key_insight.lower() for phrase in vague_phrases):
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def calculate_curation_value(self, curation_result, current_playbook) -> float:
        """
        Estimate expected value of curation (0-1).
        
        This is your existing heuristic - kept as-is.
        """
        if not curation_result.operations:
            return 0.0
        
        score = 0.0
        
        for op in curation_result.operations:
            content = op.content.lower()
            
            # Specificity
            if 'apis.' in content or 'api_docs' in content:
                score += 0.3
            
            # Actionability
            action_words = ['use', 'call', 'check', 'verify', 'ensure', 'loop', 'iterate']
            if any(word in content for word in action_words):
                score += 0.2
            
            # Avoid generic advice
            generic_phrases = ['be careful', 'think about', 'consider']
            if any(phrase in content for phrase in generic_phrases):
                score -= 0.1
        
        # Normalize
        score = score / max(len(curation_result.operations), 1)
        
        return max(0.0, min(1.0, score))
    
    def should_retry(
        self,
        attempt_num: int,
        reflection_quality: float,
        curation_value: float
    ) -> Tuple[bool, str]:
        """
        Decide whether to retry using Thompson Sampling.
        
        This replaces the old heuristic-based decision.
        """
        should_retry, reason, _ = self.policy.should_retry(
            attempt_num, reflection_quality, curation_value
        )
        return should_retry, reason
    
    def update_from_outcome(
        self,
        success: bool,
        attempt_num: int,
        reflection_quality: float = 0.0,
        curation_value: float = 0.0
    ):
        """
        Update Thompson Sampling beliefs based on task outcome.
        """
        self.policy.update_beliefs(
            success, attempt_num, reflection_quality, curation_value
        )
        
        self.attempt_history.append({
            'success': success,
            'attempt': attempt_num,
            'reflection_quality': reflection_quality,
            'curation_value': curation_value
        })
    
    def get_summary(self) -> Dict:
        """Get comprehensive summary of learning progress"""
        return {
            'policy_stats': self.policy.get_component_stats(),
            'cumulative_reward': self.cumulative_reward,
            'total_attempts': len(self.attempt_history),
            'decision_history': self.policy.decision_history[-10:]  
        }
    
    def print_summary(self):
        """Print learning summary"""
        self.policy.print_summary()