"""
Callback for integrating Extremum Seeking Control with BenchMARL experiments.
"""
from typing import List, Optional
import torch
import numpy as np
from tensordict import TensorDictBase
from benchmarl.experiment.callback import Callback
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpirical
from het_control.callbacks.callback import get_het_model
from het_control.callbacks.esc_controller import ExtremumSeekingController


class ESCCallback(Callback):
    """
    Callback that uses Extremum Seeking Control to automatically tune
    the desired SND parameter during training based on episode rewards.
    
    The ESC applies a sinusoidal perturbation to the SND continuously
    (during both training and evaluation) and uses evaluation rewards 
    to estimate gradients and update the setpoint.
    """
    
    def __init__(
        self,
        control_group: str,
        initial_snd: float,
        dither_magnitude: float = 0.1,
        dither_frequency_rad_s: float = 0.5,
        integrator_gain: float = -0.01,
        high_pass_cutoff_rad_s: float = 0.1,
        low_pass_cutoff_rad_s: float = 0.05,
        use_adaptive_gain: bool = True,
        sampling_period: float = 1.0,
        min_snd: float = 0.0,
        max_snd: float = 3.0
    ):
        """
        Args:
            control_group: Name of the agent group to control
            initial_snd: Starting value for desired SND
            dither_magnitude: Amplitude of sinusoidal perturbation
            dither_frequency_rad_s: Frequency of perturbation (rad/s)
            integrator_gain: Gain for parameter updates (negative for descent)
            high_pass_cutoff_rad_s: High-pass filter cutoff frequency (rad/s)
            low_pass_cutoff_rad_s: Low-pass filter cutoff frequency (rad/s)
            use_adaptive_gain: Whether to use adaptive gain switching
            sampling_period: Time between ESC updates (seconds)
            min_snd: Minimum allowed SND value
            max_snd: Maximum allowed SND value
        """
        super().__init__()
        self.control_group = control_group
        self.initial_snd = initial_snd
        self.min_snd = min_snd
        self.max_snd = max_snd
        
        
        # Store parameters for logging
        self.esc_params = {
            "sampling_period": sampling_period,
            "dither_frequency": dither_frequency_rad_s,
            "dither_magnitude": dither_magnitude,
            "integrator_gain": integrator_gain,
            "initial_snd": initial_snd,
            "high_pass_cutoff": high_pass_cutoff_rad_s,
            "low_pass_cutoff": low_pass_cutoff_rad_s,
            "use_adaptive_gain": use_adaptive_gain,
            "min_snd": min_snd,
            "max_snd": max_snd
        }
        
        self.model: Optional[HetControlMlpEmpirical] = None
        self.controller: Optional[ExtremumSeekingController] = None

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
                min_output=self.min_snd,
                max_output=self.max_snd
            )
            
            # Set initial desired SND (with initial perturbation)
            initial_perturbation = self.controller.a * np.sin(self.controller.phase)
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
            print(f"   SND bounds: [{self.min_snd:.1f}, {self.max_snd:.1f}]\n")
        else:
            print(f"\nWARNING: Compatible model not found for group '{self.control_group}'. Disabling ESC.\n")
            self.model = None

    def on_evaluation_end(self, rollouts: List[TensorDictBase]) -> None:
        """
        Update ESC controller based on evaluation episode rewards.
        
        The controller uses the mean reward as the performance metric (negated as cost)
        to adjust the desired SND parameter. The updated perturbed SND is immediately
        applied to the model for use in the next training phase.
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
        mean_reward = np.mean(episode_rewards)
        reward_std = np.std(episode_rewards)
        
        # ESC minimizes cost, so negate reward (maximize reward = minimize negative reward)
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
            _,                   # Duplicate gradient (not needed)
            setpoint             # SND setpoint (without perturbation)
        ) = self.controller.update(cost)
        
        # Clamp perturbed output to bounds
        perturbed_output_clamped = np.clip(perturbed_output, self.min_snd, self.max_snd)
        
        # ⭐ KEY FIX: Update model immediately with perturbed output
        # This ensures training uses the new ESC-controlled SND value
        self.model.desired_snd[:] = float(perturbed_output_clamped)
        
        # Compute actual update step
        update_step = self.model.desired_snd.item() - previous_snd
        setpoint_change = setpoint - previous_setpoint
        
        # Determine if adaptive gain was triggered
        using_high_gain = (
            self.controller.use_adaptive and 
            gradient_magnitude > self.controller.gradient_threshold
        )
        
        # Log update with more detail
        print(
            f"[ESC] Step {self.experiment.n_iters_performed:6d} | "
            f"Reward: {mean_reward:+7.3f} ±{reward_std:5.3f} | "
            f"SND: {previous_snd:.4f} → {self.model.desired_snd.item():.4f} (Δ={update_step:+.4f}) | "
            f"Setpoint: {previous_setpoint:.4f} → {setpoint:.4f} (Δ={setpoint_change:+.4f}) | "
            f"Grad: {gradient_estimate:+.5f} (RMS: {gradient_magnitude:.5f})"
            + (f" [HIGH GAIN]" if using_high_gain else "")
        )
        
        # Log comprehensive metrics
        logs = {
            # Reward metrics
            "esc/reward_mean": mean_reward,
            "esc/reward_std": reward_std,
            "esc/cost": cost,
            
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
            "esc/phase": self.controller.phase,
            "esc/m2": self.controller.m2,
            
            # Adaptive gain info
            "esc/using_high_gain": float(using_high_gain),
        }
        
        self.experiment.logger.log(logs, step=self.experiment.n_iters_performed)