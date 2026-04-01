"""
Callback for integrating PID Control with BenchMARL experiments.
"""
from typing import List, Optional
import torch
import numpy as np
from tensordict import TensorDictBase
from benchmarl.experiment.callback import Callback
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpirical
from het_control.callbacks.callback import get_het_model
from het_control.callbacks.pidController import PIDController


class PIDCallback(Callback):
    """
    Callback that uses PID Control to automatically tune
    the desired SND parameter during training based on episode rewards.
    
    The PID controller adjusts the SND parameter to maximize episode rewards
    by treating negative reward as the cost to minimize.
    """
    
    def __init__(
        self,
        control_group: str,
        initial_snd: float = 0.0,
        target_reward: float = 3.2,
        kp: float = 0.01,
        ki: float = 0.001,
        kd: float = 0.0,
        min_snd: float = 0.0,
        max_snd: float = 3.0,
        integral_clamp: float = 10.0,
        derivative_filter_alpha: float = 0.1
    ):
        """
        Args:
            control_group: Name of the agent group to control
            initial_snd: Starting value for desired SND
            target_reward: Target/maximum reward to achieve (for error calculation)
                          Use 3.2 for team reward or 1.2 for per-agent reward
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            min_snd: Minimum allowed SND value
            max_snd: Maximum allowed SND value
            integral_clamp: Maximum absolute value for integral term
            derivative_filter_alpha: Smoothing factor for derivative (0=no filter, 1=no change)
        """
        super().__init__()
        self.control_group = control_group
        self.initial_snd = initial_snd
        self.target_reward = target_reward
        self.min_snd = min_snd
        self.max_snd = max_snd
        
        # Store parameters for logging
        self.pid_params = {
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "initial_snd": initial_snd,
            "target_reward": target_reward,
            "min_snd": min_snd,
            "max_snd": max_snd,
            "integral_clamp": integral_clamp,
            "derivative_filter_alpha": derivative_filter_alpha
        }
        
        self.model: Optional[HetControlMlpEmpirical] = None
        self.controller: Optional[PIDController] = None

    def on_setup(self) -> None:
        """Initialize the controller and log hyperparameters."""
        # Log hyperparameters
        hparams = {
            "pid_control_group": self.control_group,
            **{f"pid_{k}": v for k, v in self.pid_params.items()}
        }
        self.experiment.logger.log_hparams(**hparams)
        
        # Verify control group exists
        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: PID control group '{self.control_group}' not found. Disabling controller.\n")
            return
        
        # Get model
        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)
        
        # Initialize controller if model is compatible
        if isinstance(self.model, HetControlMlpEmpirical):
            self.controller = PIDController(
                kp=self.pid_params["kp"],
                ki=self.pid_params["ki"],
                kd=self.pid_params["kd"],
                initial_value=self.initial_snd,
                target_reward=self.target_reward,
                min_output=self.min_snd,
                max_output=self.max_snd,
                integral_clamp=self.pid_params["integral_clamp"],
                derivative_filter_alpha=self.pid_params["derivative_filter_alpha"]
            )
            
            # Set initial desired SND
            self.model.desired_snd[:] = float(self.initial_snd)
            
            print(f"\n✅ SUCCESS: PID Controller initialized for group '{self.control_group}'.")
            print(f"   Initial SND: {self.initial_snd:.3f}")
            print(f"   Target reward: {self.target_reward:.3f}")
            print(f"   Kp: {self.pid_params['kp']:.4f}")
            print(f"   Ki: {self.pid_params['ki']:.4f}")
            print(f"   Kd: {self.pid_params['kd']:.4f}")
            print(f"   Integral clamp: ±{self.pid_params['integral_clamp']:.1f}")
            print(f"   Derivative filter alpha: {self.pid_params['derivative_filter_alpha']:.2f}")
            print(f"   SND bounds: [{self.min_snd:.1f}, {self.max_snd:.1f}]\n")
        else:
            print(f"\nWARNING: Compatible model not found for group '{self.control_group}'. Disabling PID.\n")
            self.model = None

    def on_evaluation_end(self, rollouts: List[TensorDictBase]) -> None:
        """
        Update PID controller based on evaluation episode rewards.
        
        The controller uses the mean reward as the performance metric (negated as cost)
        to adjust the desired SND parameter.
        """
        if self.model is None or self.controller is None:
            return
        
        # Collect episode rewards
        episode_rewards = []
        with torch.no_grad():
            for rollout in rollouts:
                reward_key = ('next', self.control_group, 'reward')
                if reward_key in rollout.keys(include_nested=True):
                    # Sum rewards over time for this episode
                    total_reward = rollout.get(reward_key).sum().item()
                    episode_rewards.append(total_reward)
        
        if not episode_rewards:
            print("\nWARNING: No episode rewards found. Skipping PID update.\n")
            return
        
        # Compute mean reward across episodes
        mean_reward = np.mean(episode_rewards)
        reward_std = np.std(episode_rewards)
        
        # Store previous SND
        previous_snd = self.model.desired_snd.item()
        
        # Update controller with the current reward
        new_snd, error, p_term, i_term, d_term = self.controller.update(mean_reward)
        
        # Update model with new SND value
        self.model.desired_snd[:] = float(new_snd)
        
        # Compute update step
        update_step = new_snd - previous_snd
        
        # Log update with detail
        print(
            f"[PID] Step {self.experiment.n_iters_performed:6d} | "
            f"Reward: {mean_reward:+7.3f} ±{reward_std:5.3f} | "
            f"Error: {error:+7.3f} (Target: {self.target_reward:.1f}) | "
            f"SND: {previous_snd:.4f} → {new_snd:.4f} (Δ={update_step:+.4f}) | "
            f"P: {p_term:+.5f}, I: {i_term:+.5f}, D: {d_term:+.5f}"
        )
        
        # Log comprehensive metrics
        logs = {
            # Reward metrics
            "pid/reward_mean": mean_reward,
            "pid/reward_std": reward_std,
            "pid/target_reward": self.target_reward,
            "pid/error": error,
            
            # SND tracking
            "pid/snd_current": new_snd,
            "pid/snd_update_step": update_step,
            
            # PID terms
            "pid/p_term": p_term,
            "pid/i_term": i_term,
            "pid/d_term": d_term,
            "pid/control_signal": p_term + i_term + d_term,
            
            # Controller state
            "pid/integral_state": self.controller.integral,
            "pid/filtered_derivative": self.controller.filtered_derivative,
        }
        
        self.experiment.logger.log(logs, step=self.experiment.n_iters_performed)