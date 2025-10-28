import logging
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from ace_appworld.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class ComponentBelief:
    """Beta distribution parameters for a component's success probability"""
    alpha: float = 1.0  # Success count (prior = 1)
    beta: float = 1.0   # Failure count (prior = 1)
    
    @property
    def mean(self) -> float:
        """Expected success probability"""
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def variance(self) -> float:
        """Uncertainty in success probability"""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
    
    def sample(self) -> float:
        """Sample from Beta distribution (Thompson Sampling)"""
        return np.random.beta(self.alpha, self.beta)


class ThompsonSamplingPolicy:
    """
    Thompson Sampling for adaptive retry decisions with component attribution.
    
    Mathematical Framework:
    ----------------------
    - Each component c ∈ {Playbook, Reflector, Curator} has reliability θ_c ~ Beta(α_c, β_c)
    - At attempt a, success probability: P(success|s) = q_R·θ_R + q_C·θ_C
    - Expected reward: EV = P(success) · r_{a+1} - cost(a)
    - Decision: RETRY if EV > 0, else SKIP
    - Update: Bayesian posterior update based on outcome
    """
    
    def __init__(
        self,
        attempt_rewards: Dict[int, float] = None,
        cost_per_attempt: float = 0.1,
        use_quality_weighted_attribution: bool = True
    ):
        """
        Args:
            attempt_rewards: Reward for success at each attempt {1: 1.0, 2: 0.6, 3: 0.3}
            cost_per_attempt: Linear cost coefficient (cost = λ·a)
            use_quality_weighted_attribution: If True, weight credit by quality scores
        """
        # Component beliefs (Beta distributions)
        self.playbook = ComponentBelief(alpha=1.0, beta=1.0)
        self.reflector = ComponentBelief(alpha=1.0, beta=1.0)
        self.curator = ComponentBelief(alpha=1.0, beta=1.0)
        
        # Reward structure
        self.attempt_rewards = attempt_rewards or {1: 1.0, 2: 0.6, 3: 0.3}
        self.cost_per_attempt = cost_per_attempt
        
        # Attribution method
        self.use_quality_weighted_attribution = use_quality_weighted_attribution
        
        # History tracking
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
        Decide whether to retry based on Thompson Sampling.
        
        Args:
            attempt_num: Current attempt number (1, 2, or 3)
            reflection_quality: Quality score q_R ∈ [0,1]
            curation_value: Quality score q_C ∈ [0,1]
            max_attempts: Maximum allowed attempts
            
        Returns:
            (should_retry, reason, debug_info)
        """
        if attempt_num >= max_attempts:
            return False, "Maximum attempts reached", {}
        
        # Sample component reliabilities (Thompson Sampling)
        theta_R = self.reflector.sample()
        theta_C = self.curator.sample()
        
        # Expected success probability
        # P(success) = q_R·θ_R + q_C·θ_C
        p_success = reflection_quality * theta_R + curation_value * theta_C
        
        # Clip to [0, 1]
        p_success = np.clip(p_success, 0.0, 1.0)
        
        # Expected reward for next attempt
        next_attempt = attempt_num + 1
        reward_if_success = self.attempt_rewards.get(next_attempt, 0.0)
        
        # Expected value calculation
        expected_reward = p_success * reward_if_success
        attempt_cost = self.cost_per_attempt * attempt_num
        
        net_expected_value = expected_reward - attempt_cost
        
        # Decision
        should_retry = net_expected_value > 0
        
        # Logging and debug info
        debug_info = {
            'sampled_theta_R': theta_R,
            'sampled_theta_C': theta_C,
            'p_success': p_success,
            'expected_reward': expected_reward,
            'attempt_cost': attempt_cost,
            'net_ev': net_expected_value,
            'reflector_belief': f"Beta({self.reflector.alpha:.2f}, {self.reflector.beta:.2f})",
            'curator_belief': f"Beta({self.curator.alpha:.2f}, {self.curator.beta:.2f})",
        }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"THOMPSON SAMPLING DECISION (Attempt {attempt_num})")
        logger.info(f"{'='*70}")
        logger.info(f"Quality Scores:")
        logger.info(f"  Reflection Quality (q_R): {reflection_quality:.3f}")
        logger.info(f"  Curation Value (q_C):     {curation_value:.3f}")
        logger.info(f"\nComponent Beliefs (learned):")
        logger.info(f"  Reflector θ_R ~ Beta({self.reflector.alpha:.2f}, {self.reflector.beta:.2f})")
        logger.info(f"    Mean: {self.reflector.mean:.3f}, Var: {self.reflector.variance:.4f}")
        logger.info(f"  Curator θ_C ~ Beta({self.curator.alpha:.2f}, {self.curator.beta:.2f})")
        logger.info(f"    Mean: {self.curator.mean:.3f}, Var: {self.curator.variance:.4f}")
        logger.info(f"\nSampled Values (this decision):")
        logger.info(f"  Sampled θ_R: {theta_R:.3f}")
        logger.info(f"  Sampled θ_C: {theta_C:.3f}")
        logger.info(f"\nExpected Outcome:")
        logger.info(f"  P(success) = q_R·θ_R + q_C·θ_C = {p_success:.3f}")
        logger.info(f"  Reward if success: {reward_if_success:.2f}")
        logger.info(f"  Expected reward: {expected_reward:.3f}")
        logger.info(f"  Attempt cost: {attempt_cost:.3f}")
        logger.info(f"  Net Expected Value: {net_expected_value:+.3f}")
        logger.info(f"\n{'─'*70}")
        logger.info(f"DECISION: {'✓ RETRY' if should_retry else '✗ SKIP'}")
        logger.info(f"{'─'*70}\n")
        
        reason = (
            f"EV={net_expected_value:+.3f} ({'>' if should_retry else '≤'} 0). "
            f"P(success)={p_success:.2f}, Reward={reward_if_success:.2f}"
        )
        
        self.decision_history.append({
            'attempt': attempt_num,
            'decision': 'RETRY' if should_retry else 'SKIP',
            **debug_info
        })
        
        return should_retry, reason, debug_info
    
    def _calculate_credit_weights(
        self,
        attempt_num: int,
        reflection_quality: float,
        curation_value: float
    ) -> Dict[str, float]:
        """
        Calculate credit attribution weights for components.
        
        Two modes:
        1. Quality-weighted: ω_c = contribution / total_contribution
        2. Fixed: Use predefined weights based on attempt number
        """
        if self.use_quality_weighted_attribution:
            # Playbook base contribution (decreases with attempts)
            playbook_base = {1: 1.0, 2: 0.2, 3: 0.1}.get(attempt_num, 0.1)
            
            # Total contribution
            total = playbook_base + reflection_quality + curation_value
            
            # Avoid division by zero
            if total < 0.01:
                total = 0.01
            
            weights = {
                'playbook': playbook_base / total,
                'reflector': reflection_quality / total,
                'curator': curation_value / total
            }
        else:
            # Fixed weights (your original intuition)
            if attempt_num == 1:
                weights = {'playbook': 1.0, 'reflector': 0.0, 'curator': 0.0}
            elif attempt_num == 2:
                weights = {'playbook': 0.0, 'reflector': 0.5, 'curator': 0.5}
            else:  # attempt_num == 3
                weights = {'playbook': 0.0, 'reflector': 0.5, 'curator': 0.5}
        
        return weights
    
    def update_beliefs(
        self,
        success: bool,
        attempt_num: int,
        reflection_quality: float = 0.0,
        curation_value: float = 0.0
    ):
        """
        Bayesian update of component beliefs based on outcome.
        
        Args:
            success: Whether the task succeeded
            attempt_num: Which attempt succeeded (or final attempt if failed)
            reflection_quality: Quality score for reflection (if used)
            curation_value: Quality score for curation (if used)
        """
        if success:
            # Calculate credit attribution
            weights = self._calculate_credit_weights(
                attempt_num, reflection_quality, curation_value
            )
            
            logger.info(f"\n{'─'*70}")
            logger.info(f"BELIEF UPDATE: SUCCESS on Attempt {attempt_num}")
            logger.info(f"{'─'*70}")
            logger.info(f"Credit Attribution:")
            logger.info(f"  Playbook:  {weights['playbook']:.3f}")
            logger.info(f"  Reflector: {weights['reflector']:.3f}")
            logger.info(f"  Curator:   {weights['curator']:.3f}")
            
            # Update beliefs with partial credit
            self.playbook.alpha += weights['playbook']
            self.playbook.beta += (1 - weights['playbook'])
            
            if attempt_num >= 2:  # Reflection was used
                self.reflector.alpha += weights['reflector']
                self.reflector.beta += (1 - weights['reflector'])
            
            if attempt_num >= 2:  # Curation was used
                self.curator.alpha += weights['curator']
                self.curator.beta += (1 - weights['curator'])
                
        else:
            # Failure: all involved components get penalized
            logger.info(f"\n{'─'*70}")
            logger.info(f"BELIEF UPDATE: FAILURE after {attempt_num} attempts")
            logger.info(f"{'─'*70}")
            
            self.playbook.beta += 1.0
            
            if attempt_num >= 2:
                self.reflector.beta += 1.0
                self.curator.beta += 1.0
        
        # Log updated beliefs
        logger.info(f"Updated Beliefs:")
        logger.info(f"  Playbook:  Beta({self.playbook.alpha:.2f}, {self.playbook.beta:.2f}) "
                   f"→ E[θ]={self.playbook.mean:.3f}")
        logger.info(f"  Reflector: Beta({self.reflector.alpha:.2f}, {self.reflector.beta:.2f}) "
                   f"→ E[θ]={self.reflector.mean:.3f}")
        logger.info(f"  Curator:   Beta({self.curator.alpha:.2f}, {self.curator.beta:.2f}) "
                   f"→ E[θ]={self.curator.mean:.3f}")
        logger.info(f"{'─'*70}\n")
        
        # Track history
        self.task_history.append({
            'success': success,
            'attempt': attempt_num,
            'playbook_belief': (self.playbook.alpha, self.playbook.beta),
            'reflector_belief': (self.reflector.alpha, self.reflector.beta),
            'curator_belief': (self.curator.alpha, self.curator.beta),
        })
    
    def get_component_stats(self) -> Dict:
        """Get current statistics for all components"""
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
        """Print a summary of learned beliefs"""
        stats = self.get_component_stats()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"THOMPSON SAMPLING POLICY SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total Tasks: {stats['total_tasks']}")
        logger.info(f"Success Rate: {stats['successful_tasks']}/{stats['total_tasks']} "
                   f"= {stats['successful_tasks']/max(stats['total_tasks'], 1):.1%}")
        logger.info(f"\nComponent Reliability (Learned):")
        
        for component in ['playbook', 'reflector', 'curator']:
            c_stats = stats[component]
            logger.info(f"\n  {component.capitalize()}:")
            logger.info(f"    E[θ] = {c_stats['mean']:.3f} (mean success probability)")
            logger.info(f"    Var[θ] = {c_stats['variance']:.4f} (uncertainty)")
            logger.info(f"    Beta({c_stats['params'][0]:.2f}, {c_stats['params'][1]:.2f})")
            logger.info(f"    Based on {c_stats['num_observations']:.0f} observations")
        
        logger.info(f"{'='*70}\n")


