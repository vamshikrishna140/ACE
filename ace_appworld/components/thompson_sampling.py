import logging
import numpy as np
from typing import Dict, Tuple, Optional
from ace_appworld.utils.logging_config import setup_logging
from ace_appworld.components.models import ComponentBelief
setup_logging()
logger = logging.getLogger(__name__)


class ThompsonSamplingPolicy:
    """
    Thompson Sampling for adaptive retries with component attribution.
    
    Basic idea: each component (Playbook, Reflector, Curator) has a reliability score
    that we learn over time. We sample from these beliefs to decide if retrying is worth it.
    """
    
    def __init__(
        self,
        attempt_rewards: Dict[int, float] = None,
        cost_per_attempt: float = 0.1,
        use_quality_weighted_attribution: bool = True
    ):
        # Start with weak priors - beta(1,1) is uniform
        self.playbook = ComponentBelief(alpha=1.0, beta=1.0)
        self.reflector = ComponentBelief(alpha=1.0, beta=1.0)
        self.curator = ComponentBelief(alpha=1.0, beta=1.0)
        
        # Reward schedule - decreases with attempts
        self.attempt_rewards = attempt_rewards or {1: 1.0, 2: 0.6, 3: 0.3}
        self.cost_per_attempt = cost_per_attempt
        
        self.use_quality_weighted_attribution = use_quality_weighted_attribution
        
        # Track history for debugging
        self.task_history = []
        self.decision_history = []
        
    def should_retry(
        self,
        attempt_num: int,
        reflection_quality: float,
        curation_value: float,
        max_attempts: int = 3
    ) -> Tuple[bool, str, Dict]:
        """
        Main decision function - should we retry or give up?
        
        We sample component reliabilities and compute expected value.
        Retry if EV > 0.
        """
        if attempt_num >= max_attempts:
            return False, "max attempts reached", {}
        
        # Thompson sampling: draw from current beliefs
        theta_R = self.reflector.sample()
        theta_C = self.curator.sample()
        
        # Expected success prob = weighted sum of component contributions
        # This is a simple linear model, could be fancier
        p_success = reflection_quality * theta_R + curation_value * theta_C
        p_success = np.clip(p_success, 0.0, 1.0)
        
        # EV calculation
        next_attempt = attempt_num + 1
        reward_if_success = self.attempt_rewards.get(next_attempt, 0.0)
        
        expected_reward = p_success * reward_if_success
        attempt_cost = self.cost_per_attempt * attempt_num
        
        net_ev = expected_reward - attempt_cost
        
        should_retry = net_ev > 0
        
        # Log decision rationale
        debug_info = {
            'sampled_theta_R': theta_R,
            'sampled_theta_C': theta_C,
            'p_success': p_success,
            'expected_reward': expected_reward,
            'attempt_cost': attempt_cost,
            'net_ev': net_ev,
            'reflector_belief': f"Beta({self.reflector.alpha:.2f}, {self.reflector.beta:.2f})",
            'curator_belief': f"Beta({self.curator.alpha:.2f}, {self.curator.beta:.2f})",
        }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Thompson Sampling Decision (attempt {attempt_num})")
        logger.info(f"  Quality scores: q_R={reflection_quality:.3f}, q_C={curation_value:.3f}")
        logger.info(f"  Sampled beliefs: theta_R={theta_R:.3f}, theta_C={theta_C:.3f}")
        logger.info(f"  P(success) = {p_success:.3f}")
        logger.info(f"  Expected reward = {expected_reward:.3f}, Cost = {attempt_cost:.3f}")
        logger.info(f"  Net EV = {net_ev:+.3f}")
        logger.info(f"  Decision: {'RETRY' if should_retry else 'SKIP'}")
        logger.info(f"{'='*70}\n")
        
        reason = f"EV={net_ev:+.3f}, P(success)={p_success:.2f}"
        
        self.decision_history.append({
            'attempt': attempt_num,
            'decision': 'RETRY' if should_retry else 'SKIP',
            **debug_info
        })
        
        return should_retry, reason, debug_info
    
    def _calc_credit_weights(
        self,
        attempt_num: int,
        reflection_quality: float,
        curation_value: float
    ):
        """
        How much credit should each component get for this outcome?
        
        Two modes:
        - Quality-weighted: proportional to contribution
        - Fixed: hardcoded based on attempt number
        """
        if self.use_quality_weighted_attribution:
            # Playbook contribution decreases over attempts
            playbook_base = {1: 1.0, 2: 0.2, 3: 0.1}.get(attempt_num, 0.1)
            
            total = playbook_base + reflection_quality + curation_value
            if total < 0.01:  # avoid div by zero
                total = 0.01
            
            return {
                'playbook': playbook_base / total,
                'reflector': reflection_quality / total,
                'curator': curation_value / total
            }
        else:
            # Simple fixed schedule
            if attempt_num == 1:
                return {'playbook': 1.0, 'reflector': 0.0, 'curator': 0.0}
            else:
                return {'playbook': 0.0, 'reflector': 0.5, 'curator': 0.5}
    
    def update_beliefs(
        self,
        success: bool,
        attempt_num: int,
        reflection_quality: float = 0.0,
        curation_value: float = 0.0
    ):
        """
        Bayesian update after observing outcome.
        Success -> increase alpha, Failure -> increase beta
        """
        if success:
            weights = self._calc_credit_weights(
                attempt_num, reflection_quality, curation_value
            )
            
            logger.info(f"Belief update: SUCCESS on attempt {attempt_num}")
            logger.info(f"  Credit weights: playbook={weights['playbook']:.3f}, "
                       f"reflector={weights['reflector']:.3f}, curator={weights['curator']:.3f}")
            
            # Fractional updates based on credit
            self.playbook.alpha += weights['playbook']
            self.playbook.beta += (1 - weights['playbook'])
            
            if attempt_num >= 2:
                self.reflector.alpha += weights['reflector']
                self.reflector.beta += (1 - weights['reflector'])
                
                self.curator.alpha += weights['curator']
                self.curator.beta += (1 - weights['curator'])
                
        else:
            logger.info(f"Belief update: FAILURE after {attempt_num} attempts")
            
            # Everyone involved gets penalized
            self.playbook.beta += 1.0
            
            if attempt_num >= 2:
                self.reflector.beta += 1.0
                self.curator.beta += 1.0
        
        # Show updated beliefs
        logger.info(f"  Updated beliefs:")
        logger.info(f"    Playbook: Beta({self.playbook.alpha:.2f}, {self.playbook.beta:.2f}), "
                   f"mean={self.playbook.mean:.3f}")
        logger.info(f"    Reflector: Beta({self.reflector.alpha:.2f}, {self.reflector.beta:.2f}), "
                   f"mean={self.reflector.mean:.3f}")
        logger.info(f"    Curator: Beta({self.curator.alpha:.2f}, {self.curator.beta:.2f}), "
                   f"mean={self.curator.mean:.3f}\n")
        
        self.task_history.append({
            'success': success,
            'attempt': attempt_num,
            'playbook_belief': (self.playbook.alpha, self.playbook.beta),
            'reflector_belief': (self.reflector.alpha, self.reflector.beta),
            'curator_belief': (self.curator.alpha, self.curator.beta),
        })
    
    def get_component_stats(self):
        """Get current stats for all components"""
        return {
            'playbook': {
                'mean': self.playbook.mean,
                'variance': self.playbook.variance,
                'params': (self.playbook.alpha, self.playbook.beta),
                'num_observations': self.playbook.alpha + self.playbook.beta - 2
            },
            'reflector': {
                'mean': self.reflector.mean,
                'variance': self.reflector.variance,
                'params': (self.reflector.alpha, self.reflector.beta),
                'num_observations': self.reflector.alpha + self.reflector.beta - 2
            },
            'curator': {
                'mean': self.curator.mean,
                'variance': self.curator.variance,
                'params': (self.curator.alpha, self.curator.beta),
                'num_observations': self.curator.alpha + self.curator.beta - 2
            },
            'total_tasks': len(self.task_history),
            'successful_tasks': sum(1 for t in self.task_history if t['success'])
        }
    
    def print_summary(self):
        """Print summary of what we've learned"""
        stats = self.get_component_stats()
        
        print(f"\n{'='*70}")
        print(f"Thompson Sampling Summary")
        print(f"{'='*70}")
        print(f"Total tasks: {stats['total_tasks']}")
        print(f"Success rate: {stats['successful_tasks']}/{stats['total_tasks']}")
        print(f"\nLearned component reliabilities:")
        
        for comp in ['playbook', 'reflector', 'curator']:
            c = stats[comp]
            print(f"\n  {comp.capitalize()}:")
            print(f"    Mean reliability: {c['mean']:.3f}")
            print(f"    Uncertainty (var): {c['variance']:.4f}")
            print(f"    Beta({c['params'][0]:.2f}, {c['params'][1]:.2f})")
            print(f"    Based on {c['num_observations']:.0f} observations")
        
        print(f"{'='*70}\n")