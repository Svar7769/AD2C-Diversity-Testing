"""
Callback for integrating Extremum Seeking Control with BenchMARL experiments.
Includes state machine for detecting stability and switching between active ESC and steady-state modes.
"""
from typing import List, Optional
from enum import Enum
from collections import deque
import torch
import numpy as np
from tensordict import TensorDictBase
from benchmarl.experiment.callback import Callback
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpirical
from het_control.callbacks.callback import get_het_model
from het_control.callbacks.esc_controller import ExtremumSeekingController


class ESCState(Enum):
    """States for the ESC state machine."""
    ACTIVE = "active"      # ESC is actively adjusting SND
    STABLE = "stable"      # System is stable, SND held constant


class AdaptiveESCCallback(Callback):
    """
    Callback that uses Extremum Seeking Control to automatically tune
    the desired SND parameter during training based on episode rewards.
    
    The ESC applies a sinusoidal perturbation to the SND continuously
    (during both training and evaluation) and uses evaluation rewards 
    to estimate gradients and update the setpoint.
    
    State Machine:
    - ACTIVE: ESC actively adjusts SND based on gradient estimates
    - STABLE: SND is held constant when system reaches stable state
    
    Transitions:
    - ACTIVE → STABLE: When setpoint and reward are stable for stability_window episodes
    - STABLE → ACTIVE: When reward fluctuates significantly (exceeds stability_threshold)
    """
    
    def __init__(
        self,
        control_group: str,
        initial_snd: float,
        dither_magnitude: float = 0.2,
        dither_frequency: float = 1.0,
        integrator_gain: float = -0.05,
        high_pass_cutoff: float = 1.0,
        low_pass_cutoff: float = 1.0,
        use_adaptive_gain: bool = True,
        sampling_period: float = 1.0,
        min_snd: float = 0.0,
        max_snd: float = 3.0,
        # State machine parameters
        stability_window: int = 5,
        setpoint_stability_threshold: float = 0.02,
        reward_stability_threshold: float = 0.05,
        reward_change_threshold: float = 0.10,
        min_episodes_before_stable: int = 15,
        gradient_stability_threshold: float = 0.5
    ):
        """
        Args:
            control_group: Name of the agent group to control
            initial_snd: Starting value for desired SND
            dither_magnitude: Amplitude of sinusoidal perturbation
            dither_frequency: Frequency of perturbation (rad/s)
            integrator_gain: Gain for parameter updates (negative for descent)
            high_pass_cutoff: High-pass filter cutoff frequency (rad/s)
            low_pass_cutoff: Low-pass filter cutoff frequency (rad/s)
            use_adaptive_gain: Whether to use adaptive gain switching
            sampling_period: Time between ESC updates (seconds)
            min_snd: Minimum allowed SND value
            max_snd: Maximum allowed SND value
            stability_window: Number of episodes to check for stability (default: 20)
            setpoint_stability_threshold: Max std dev of setpoint changes to be considered stable (default: 0.005)
            reward_stability_threshold: Max relative std dev of rewards to be considered stable (default: 0.05)
            reward_change_threshold: Relative reward change that triggers exit from STABLE state (default: 0.2)
            min_episodes_before_stable: Minimum episodes before allowing STABLE transition (default: 30)
            gradient_stability_threshold: Max RMS gradient magnitude to be considered stable (default: 0.01)
        """
        super().__init__()
        self.control_group = control_group
        self.initial_snd = initial_snd
        self.min_snd = min_snd
        self.max_snd = max_snd
        
        # State machine parameters
        self.stability_window = stability_window
        self.setpoint_stability_threshold = setpoint_stability_threshold
        self.reward_stability_threshold = reward_stability_threshold
        self.reward_change_threshold = reward_change_threshold
        self.min_episodes_before_stable = min_episodes_before_stable
        self.gradient_stability_threshold = gradient_stability_threshold
        
        # Store parameters for logging
        self.esc_params = {
            "sampling_period": sampling_period,
            "dither_frequency": dither_frequency,
            "dither_magnitude": dither_magnitude,
            "integrator_gain": integrator_gain,
            "initial_snd": initial_snd,
            "high_pass_cutoff": high_pass_cutoff,
            "low_pass_cutoff": low_pass_cutoff,
            "use_adaptive_gain": use_adaptive_gain,
            "min_snd": min_snd,
            "max_snd": max_snd,
            "stability_window": stability_window,
            "setpoint_stability_threshold": setpoint_stability_threshold,
            "reward_stability_threshold": reward_stability_threshold,
            "reward_change_threshold": reward_change_threshold,
            "min_episodes_before_stable": min_episodes_before_stable,
            "gradient_stability_threshold": gradient_stability_threshold
        }
        
        self.model: Optional[HetControlMlpEmpirical] = None
        self.controller: Optional[ExtremumSeekingController] = None
        
        # State machine
        self.state = ESCState.ACTIVE
        self.setpoint_history = deque(maxlen=stability_window)
        self.reward_history = deque(maxlen=stability_window)
        self.gradient_history = deque(maxlen=stability_window)
        self.stable_snd: Optional[float] = None  # SND value to hold when in STABLE state
        self.stable_reward_baseline: Optional[float] = None  # Reward baseline in STABLE state
        self.episodes_in_stable = 0
        self.total_episodes = 0  # Track total episodes for min_episodes_before_stable

    def on_setup(self) -> None:
        """Initialize the controller and log hyperparameters."""
        # Log hyperparameters
        hparams = {
            "esc_control_group": self.control_group,
            **{f"esc_{k}": v for k, v in self.esc_params.items()}
        }
        self.experiment.logger.log_hparams(**hparams)
        
        # Verify control group exists
        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: ESC control group '{self.control_group}' not found. Disabling controller.\n")
            return
        
        # Get model
        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)
        
        # Initialize controller if model is compatible
        if isinstance(self.model, HetControlMlpEmpirical):
            self.controller = ExtremumSeekingController(
                sampling_period=self.esc_params["sampling_period"],
                dither_frequency=self.esc_params["dither_frequency"],
                dither_magnitude=self.esc_params["dither_magnitude"],
                integrator_gain=self.esc_params["integrator_gain"],
                initial_value=self.initial_snd,
                high_pass_cutoff=self.esc_params["high_pass_cutoff"],
                low_pass_cutoff=self.esc_params["low_pass_cutoff"],
                use_adaptive_gain=self.esc_params["use_adaptive_gain"],
                min_output=self.min_snd
            )
            
            # Set initial desired SND (with initial perturbation)
            initial_perturbation = self.controller.a * np.sin(self.controller.wt)
            initial_output = np.clip(
                self.initial_snd + initial_perturbation,
                self.min_snd,
                self.max_snd
            )
            self.model.desired_snd[:] = float(initial_output)
            
            print(f"\n✅ SUCCESS: ESC Controller initialized for group '{self.control_group}'.")
            print(f"   Initial SND: {self.initial_snd:.3f}")
            print(f"   Dither: ±{self.esc_params['dither_magnitude']:.3f} @ {self.esc_params['dither_frequency']:.2f} rad/s")
            print(f"   Integrator gain: {self.esc_params['integrator_gain']:.4f}")
            print(f"   High-pass cutoff: {self.esc_params['high_pass_cutoff']:.3f} rad/s")
            print(f"   Low-pass cutoff: {self.esc_params['low_pass_cutoff']:.3f} rad/s")
            print(f"   Adaptive gain: {self.esc_params['use_adaptive_gain']}")
            print(f"   SND bounds: [{self.min_snd:.1f}, {self.max_snd:.1f}]")
            print(f"   State Machine:")
            print(f"     - Stability window: {self.stability_window} episodes")
            print(f"     - Min episodes before stable: {self.min_episodes_before_stable}")
            print(f"     - Setpoint stability: {self.setpoint_stability_threshold:.5f}")
            print(f"     - Reward stability: {self.reward_stability_threshold:.4f}")
            print(f"     - Gradient stability: {self.gradient_stability_threshold:.5f}")
            print(f"     - Reward change threshold: {self.reward_change_threshold:.4f}\n")
        else:
            print(f"\nWARNING: Compatible model not found for group '{self.control_group}'. Disabling ESC.\n")
            self.model = None

    def _check_stability(self) -> bool:
        """
        Check if the system is stable based on setpoint, reward, and gradient history.
        
        Returns:
            True if system is stable (low variance in setpoint, reward, and gradient)
        """
        # Must have enough episodes in history AND meet minimum total episodes
        if len(self.setpoint_history) < self.stability_window:
            return False
        
        if self.total_episodes < self.min_episodes_before_stable:
            return False
        
        # Check setpoint stability (low std dev of changes)
        setpoint_changes = np.diff(list(self.setpoint_history))
        setpoint_std = np.std(setpoint_changes)
        setpoint_stable = setpoint_std < self.setpoint_stability_threshold
        
        # Check reward stability (low relative std dev)
        rewards = (np.array(list(self.reward_history)))
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        reward_relative_std = reward_std / (abs(reward_mean) + 1e-8)
        reward_stable = reward_relative_std < self.reward_stability_threshold
        
        # Check gradient stability (low RMS gradient magnitude)
        gradients = np.array(list(self.gradient_history))
        gradient_rms = np.sqrt(np.mean(gradients**2))
        gradient_stable = gradient_rms < self.gradient_stability_threshold
        
        # All conditions must be met
        is_stable = setpoint_stable and reward_stable and gradient_stable
        
        # Log stability metrics for debugging
        if len(self.setpoint_history) == self.stability_window and self.total_episodes % 5 == 0:
            print(f"    [Stability Check] Episodes: {self.total_episodes}/{self.min_episodes_before_stable} | "
                  f"Setpoint std: {setpoint_std:.5f} ({'✓' if setpoint_stable else '✗'}) | "
                  f"Reward rel std: {reward_relative_std:.4f} ({'✓' if reward_stable else '✗'}) | "
                  f"Gradient RMS: {gradient_rms:.5f} ({'✓' if gradient_stable else '✗'})")
        
        return is_stable

    def _check_reward_change(self, current_reward: float) -> bool:
        """
        Check if reward has changed significantly from the stable baseline.
        
        Args:
            current_reward: Current episode reward
            
        Returns:
            True if reward has changed beyond threshold
        """
        if self.stable_reward_baseline is None:
            return False
        
        comparison_reward = np.mean(list(self.reward_history)) if len(self.reward_history) > 0 else current_reward
        
        diff = comparison_reward - self.stable_reward_baseline
        
        rel_change = diff / (abs(self.stable_reward_baseline) + 1e-8)
        
        if rel_change < -self.reward_change_threshold:
            return True
        
        if rel_change > (self.reward_change_threshold * 2):
            return True
        
        return False
    
    def on_evaluation_end(self, rollouts: List[TensorDictBase]) -> None:
        """
        Update ESC controller based on evaluation episode rewards.
        
        State machine handles transitions between ACTIVE and STABLE states.
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
            print("\nWARNING: No episode rewards found. Skipping ESC update.\n")
            return
        
        # Compute mean reward across episodes
        episode_rewards = episode_rewards
        mean_reward = np.mean(episode_rewards) 
        reward_std = np.std(episode_rewards)
        
        # Add to reward history
        self.reward_history.append(mean_reward)
        
        # Increment episode counter
        self.total_episodes += 1
        
        # Store previous state for logging
        previous_state = self.state
        
        # State machine logic
        if self.state == ESCState.ACTIVE:
            # ESC minimizes cost, so negate reward
            cost = -mean_reward
            
            # Store previous SND (actual value with perturbation)
            previous_snd = self.model.desired_snd.item()
            previous_setpoint = self.controller.theta_0 + self.controller.integral
            
            # Update controller with the cost
            (
                perturbed_output,    # SND with perturbation (for next iteration)
                hpf_output,          # High-pass filtered cost
                gradient_estimate,   # Gradient estimate (LPF output)
                gradient_magnitude,  # RMS of gradient
                demodulated,         # Demodulated signal
                setpoint             # SND setpoint (without perturbation)
            ) = self.controller.update(cost)
            
            # Add setpoint to history
            self.setpoint_history.append(setpoint)
            
            # Add gradient magnitude to history
            self.gradient_history.append(gradient_magnitude)
            
            # Clamp perturbed output to bounds
            perturbed_output_clamped = np.clip(perturbed_output, self.min_snd, self.max_snd)
            
            # Apply the new ESC-controlled SND value
            self.model.desired_snd[:] = float(perturbed_output_clamped)
            
            # Compute actual update step
            update_step = self.model.desired_snd.item() - previous_snd
            setpoint_change = setpoint - previous_setpoint
            
            # Determine if adaptive gain was triggered
            using_high_gain = (
                self.controller.use_adaptive and 
                gradient_magnitude > self.controller.gradient_threshold
            )
            
            # Check for stability transition
            if self._check_stability():
                self.state = ESCState.STABLE
                self.stable_snd = setpoint  # Hold setpoint (no perturbation in stable state)
                self.stable_reward_baseline = mean_reward
                self.episodes_in_stable = 0
                
                # Compute final stability metrics for logging
                setpoint_changes = np.diff(list(self.setpoint_history))
                setpoint_std = np.std(setpoint_changes)
                rewards = np.array(list(self.reward_history))
                reward_relative_std = np.std(rewards) / (abs(np.mean(rewards)) + 1e-8)
                gradients = np.array(list(self.gradient_history))
                gradient_rms = np.sqrt(np.mean(gradients**2))
                
                print(f"\n{'='*80}")
                print(f"STATE TRANSITION: ACTIVE → STABLE (Episode {self.total_episodes})")
                print(f"  Setpoint stabilized at: {self.stable_snd:.4f} (std of changes: {setpoint_std:.6f})")
                print(f"  Reward baseline: {self.stable_reward_baseline:.3f} (rel std: {reward_relative_std:.4f})")
                print(f"  Gradient RMS: {gradient_rms:.6f}")
                print(f"{'='*80}\n")
            
            # Log update with state info
            print(
                f"[ESC-{self.state.value.upper()}] Step {self.experiment.n_iters_performed:6d} (Ep {self.total_episodes}) | "
                f"Reward: {mean_reward:+7.3f} ±{reward_std:5.3f} | "
                f"SND: {previous_snd:.4f} → {self.model.desired_snd.item():.4f} (Δ={update_step:+.4f}) | "
                f"Setpoint: {previous_setpoint:.4f} → {setpoint:.4f} (Δ={setpoint_change:+.4f}) | "
                f"Grad: {gradient_estimate:+.5f} (RMS: {gradient_magnitude:.5f})"
                + (f" [HIGH GAIN]" if using_high_gain else "")
            )
            
            # Log comprehensive metrics
            logs = {
                # State machine
                "esc/state": 0.0,  # 0 = ACTIVE
                "esc/episodes_in_stable": 0,
                "esc/total_episodes": self.total_episodes,
                
                # Reward metrics
                "esc/reward_mean": mean_reward,
                "esc/reward_std": reward_std,
                "esc/cost": cost,
                "esc/lpf_input": demodulated,
                
                # SND tracking (actual applied values)
                "esc/snd_actual": self.model.desired_snd.item(),
                "esc/snd_actual_previous": previous_snd,
                "esc/snd_update_step": update_step,
                
                # Setpoint tracking (without perturbation)
                "esc/snd_setpoint": setpoint,
                "esc/snd_setpoint_previous": previous_setpoint,
                "esc/snd_setpoint_change": setpoint_change,
                
                # Perturbation info
                "esc/perturbation_current": perturbed_output_clamped - setpoint,
                "esc/perturbation_raw": perturbed_output - setpoint,
                
                # Controller internals
                "esc/gradient_estimate": gradient_estimate,
                "esc/gradient_magnitude": gradient_magnitude,
                "esc/hpf_output": hpf_output,
                "esc/integrator_state": self.controller.integral,
                "esc/phase": self.controller.wt,
                "esc/m2": self.controller.m2,
                
                # Adaptive gain info
                "esc/using_high_gain": float(using_high_gain),
            }
            
        elif self.state == ESCState.STABLE:
            # In stable state, hold SND constant (no perturbation)
            self.episodes_in_stable += 1
            
            # Keep SND at the stable setpoint
            self.model.desired_snd[:] = float(self.stable_snd)
            
            # Check if reward has fluctuated significantly
            if self._check_reward_change(mean_reward):
                self.state = ESCState.ACTIVE
                # Reset controller to current stable setpoint
                self.controller.integral = self.stable_snd - self.controller.theta_0
                # Clear histories to avoid stale data
                self.setpoint_history.clear()
                self.reward_history.clear()
                self.reward_history.append(mean_reward)
                print(f"\n{'='*80}")
                print(f"STATE TRANSITION: STABLE → ACTIVE")
                print(f"  Reward changed significantly: {self.stable_reward_baseline:.3f} → {mean_reward:.3f}")
                print(f"  Relative change: {abs(mean_reward - self.stable_reward_baseline) / (abs(self.stable_reward_baseline) + 1e-8):.3f}")
                print(f"  Resuming ESC from setpoint: {self.stable_snd:.4f}")
                print(f"{'='*80}\n")
            
            # Log stable state
            print(
                f"[ESC-{self.state.value.upper()}] Step {self.experiment.n_iters_performed:6d} (Ep {self.total_episodes}) | "
                f"Reward: {mean_reward:+7.3f} ±{reward_std:5.3f} | "
                f"SND: {self.stable_snd:.4f} (HELD) | "
                f"Episodes in stable: {self.episodes_in_stable} | "
                f"Reward Δ from baseline: {mean_reward - self.stable_reward_baseline:+.3f}"
            )
            
            logs = {
                # State machine
                "esc/state": 1.0,  # 1 = STABLE
                "esc/episodes_in_stable": self.episodes_in_stable,
                "esc/total_episodes": self.total_episodes,
                
                # Reward metrics
                "esc/reward_mean": mean_reward,
                "esc/reward_std": reward_std,
                "esc/reward_baseline": self.stable_reward_baseline,
                "esc/reward_deviation": mean_reward - self.stable_reward_baseline,
                
                # SND tracking
                "esc/snd_actual": self.stable_snd,
                "esc/snd_setpoint": self.stable_snd,
                "esc/snd_update_step": 0.0,
                "esc/snd_setpoint_change": 0.0,
                
                # Zero out controller metrics in stable state
                "esc/gradient_estimate": 0.0,
                "esc/gradient_magnitude": 0.0,
                "esc/perturbation_current": 0.0,
            }
        
        # Log state transition if occurred
        if previous_state != self.state:
            logs["esc/state_transition"] = 1.0
        else:
            logs["esc/state_transition"] = 0.0
        
        self.experiment.logger.log(logs, step=self.experiment.n_iters_performed)