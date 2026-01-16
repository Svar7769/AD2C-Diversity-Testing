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


class AdaptiveESCCallback(Callback):
    """
    Callback that uses Extremum Seeking Control to automatically tune
    the desired SND parameter during training based on episode rewards.
    
    The ESC applies a sinusoidal perturbation to the SND continuously
    (during both training and evaluation) and uses evaluation rewards 
    to estimate gradients and update the setpoint.
    
    Supports both classical ESC and adaptive extensions (RMSprop-style gain,
    adaptive dither) that can be toggled via flags.
    """
    
    def __init__(
        self,
        control_group: str,
        initial_snd: float,
        dither_magnitude: float = 0.1,
        dither_frequency_rad_s: float = 0.5,
        integrator_gain: float = -0.01,
        high_pass_cutoff_rad_s: float = 0.01,
        low_pass_cutoff_rad_s: float = 0.1,
        use_adaptive_gain: bool = False,
        use_adaptive_dither: bool = False,
        sampling_period: float = 1.0,
        min_snd: float = 0.0,
        max_snd: float = 3.0,
        # Adaptive gain parameters
        gain_adaptation_mode: str = "rmsprop",
        binary_gain_threshold: float = 0.2,
        binary_high_gain_multiplier: float = 2.5,
        # Adaptive dither parameters
        dither_decay_rate: float = 0.999,
        min_dither_ratio: float = 0.1,
        dither_boost_threshold: float = 0.01,
        dither_boost_rate: float = 1.02,
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
            use_adaptive_gain: Whether to use adaptive gain adjustment
            use_adaptive_dither: Whether to use adaptive dither magnitude
            sampling_period: Time between ESC updates (seconds)
            min_snd: Minimum allowed SND value
            max_snd: Maximum allowed SND value
            
            # Adaptive Gain Parameters:
            gain_adaptation_mode: Mode for adaptive gain ("rmsprop", "binary", "gradient_norm")
            binary_gain_threshold: Threshold for binary mode switching
            binary_high_gain_multiplier: Multiplier for high gain in binary mode
            
            # Adaptive Dither Parameters:
            dither_decay_rate: Rate at which dither decays (0.995-0.999)
            min_dither_ratio: Minimum dither as fraction of initial (e.g., 0.1 = 10%)
            dither_boost_threshold: Gradient magnitude below which to boost dither
            dither_boost_rate: Rate at which dither increases when boosted
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
            "use_adaptive_dither": use_adaptive_dither,
            "gain_adaptation_mode": gain_adaptation_mode,
            "binary_gain_threshold": binary_gain_threshold,
            "binary_high_gain_multiplier": binary_high_gain_multiplier,
            "dither_decay_rate": dither_decay_rate,
            "min_dither_ratio": min_dither_ratio,
            "dither_boost_threshold": dither_boost_threshold,
            "dither_boost_rate": dither_boost_rate,
            "min_snd": min_snd,
            "max_snd": max_snd,
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
            try:
                self.controller = ExtremumSeekingController(
                    sampling_period=self.esc_params["sampling_period"],
                    dither_frequency=self.esc_params["dither_frequency"],
                    dither_magnitude=self.esc_params["dither_magnitude"],
                    integrator_gain=self.esc_params["integrator_gain"],
                    initial_value=self.initial_snd,
                    high_pass_cutoff=self.esc_params["high_pass_cutoff"],
                    low_pass_cutoff=self.esc_params["low_pass_cutoff"],
                    use_adaptive_gain=self.esc_params["use_adaptive_gain"],
                    use_adaptive_dither=self.esc_params["use_adaptive_dither"],
                    min_output=self.min_snd,
                    gain_adaptation_mode=self.esc_params["gain_adaptation_mode"],
                    binary_gain_threshold=self.esc_params["binary_gain_threshold"],
                    binary_high_gain_multiplier=self.esc_params["binary_high_gain_multiplier"],
                    dither_decay_rate=self.esc_params["dither_decay_rate"],
                    min_dither_ratio=self.esc_params["min_dither_ratio"],
                    dither_boost_threshold=self.esc_params["dither_boost_threshold"],
                    dither_boost_rate=self.esc_params["dither_boost_rate"],
                )
            except ValueError as e:
                print(f"\n❌ ERROR: Failed to initialize ESC Controller: {e}\n")
                self.model = None
                return
            
            # Set initial desired SND (with initial perturbation)
            initial_perturbation = self.controller.a * np.sin(self.controller.phase)
            initial_output = np.clip(
                self.initial_snd + initial_perturbation,
                self.min_snd,
                self.max_snd
            )
            self.model.desired_snd[:] = float(initial_output)
            
            # Print configuration
            mode_desc = "Classical ESC"
            if self.esc_params["use_adaptive_gain"] or self.esc_params["use_adaptive_dither"]:
                mode_desc = "Adaptive ESC ("
                features = []
                if self.esc_params["use_adaptive_gain"]:
                    features.append(f"{self.esc_params['gain_adaptation_mode']} gain")
                if self.esc_params["use_adaptive_dither"]:
                    features.append("adaptive dither")
                mode_desc += ", ".join(features) + ")"
            
            print(f"\n✅ SUCCESS: ESC Controller initialized for group '{self.control_group}'.")
            print(f"   Mode: {mode_desc}")
            print(f"   Initial SND: {self.initial_snd:.3f}")
            print(f"   Dither: ±{self.esc_params['dither_magnitude']:.3f} @ {self.esc_params['dither_frequency']:.2f} rad/s")
            print(f"   Integrator gain: {self.esc_params['integrator_gain']:.4f}")
            print(f"   High-pass cutoff: {self.esc_params['high_pass_cutoff']:.3f} rad/s")
            print(f"   Low-pass cutoff: {self.esc_params['low_pass_cutoff']:.3f} rad/s")
            print(f"   Frequency ordering: ωh={self.esc_params['high_pass_cutoff']:.3f} < ωl={self.esc_params['low_pass_cutoff']:.2f} < ω={self.esc_params['dither_frequency']:.2f} ✓")
            if self.esc_params["use_adaptive_gain"]:
                print(f"   Adaptive gain mode: {self.esc_params['gain_adaptation_mode']}")
            if self.esc_params["use_adaptive_dither"]:
                print(f"   Dither decay: {self.esc_params['dither_decay_rate']:.4f}, min ratio: {self.esc_params['min_dither_ratio']:.2f}")
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
        
        # Store previous state
        previous_snd = self.model.desired_snd.item()
        previous_setpoint = self.controller.theta_0 + self.controller.integral
        previous_dither = self.controller.a
        
        # Update controller with the cost
        (
            perturbed_output,    # SND with perturbation (for next iteration)
            hpf_output,          # High-pass filtered cost
            gradient_estimate,   # Gradient estimate (LPF output)
            gradient_magnitude,  # RMS of gradient
            _,                   # Duplicate gradient (not needed)
            setpoint             # SND setpoint (without perturbation)
        ) = self.controller.update(cost)
        
        # Get controller state for logging
        controller_state = self.controller.get_state()
        
        # Clamp perturbed output to bounds
        perturbed_output_clamped = np.clip(perturbed_output, self.min_snd, self.max_snd)
        
        # Update model immediately with perturbed output
        self.model.desired_snd[:] = float(perturbed_output_clamped)
        
        # Compute changes
        update_step = self.model.desired_snd.item() - previous_snd
        setpoint_change = setpoint - previous_setpoint
        dither_change = self.controller.a - previous_dither
        
        # Build console log message (NEW FORMAT - matching expected output)
        log_msg = (
            f"[ESC] Updated SND: {self.model.desired_snd.item():.4f} "
            f"(Reward: {mean_reward:+.3f}, Update Step: {update_step:+.4f})"
        )
        
        # Add adaptive features to log if enabled
        if self.controller.use_adaptive_gain:
            gain_used = self.controller.current_gain
            log_msg += f" | Gain: {gain_used:.5f}"
            
        if self.controller.use_adaptive_dither:
            dither_ratio = self.controller.a / self.controller.a_initial
            log_msg += f" | Dither: {self.controller.a:.4f} ({dither_ratio:.1%})"
            if controller_state["stuck_counter"] > 0:
                log_msg += f" [STUCK:{controller_state['stuck_counter']}]"
        
        print(log_msg)
        
        # Comprehensive logging (matching existing log names)
        logs = {
            # Core metrics (matching existing names for backward compatibility)
            "esc/mean_reward": mean_reward,
            "esc/cost": cost,
            "esc/diversity_output": perturbed_output_clamped,
            "esc/diversity_setpoint": setpoint,
            "esc/gradient_estimate": gradient_estimate,
            "esc/hpf_output": hpf_output,
            "esc/lpf_output": gradient_estimate,
            "esc/m2_sqrt": gradient_magnitude,
            "esc/update_step": update_step,
            
            # Additional useful metrics
            "esc/reward_std": reward_std,
            "esc/snd_previous": previous_snd,
            "esc/setpoint_change": setpoint_change,
            "esc/perturbation_current": perturbed_output_clamped - setpoint,
            "esc/integrator_state": self.controller.integral,
            "esc/phase": self.controller.phase,
        }
        
        # Add adaptive gain metrics if enabled
        if self.controller.use_adaptive_gain:
            logs.update({
                "esc/gain_current": self.controller.current_gain,
                "esc/gain_base": self.controller.k,
                "esc/gain_ratio": abs(self.controller.current_gain / self.controller.k),
            })
        
        # Add adaptive dither metrics if enabled
        if self.controller.use_adaptive_dither:
            logs.update({
                "esc/dither_amplitude": self.controller.a,
                "esc/dither_amplitude_initial": self.controller.a_initial,
                "esc/dither_ratio": controller_state["dither_ratio"],
                "esc/dither_change": dither_change,
                "esc/cost_variance": controller_state["cost_variance"],
                "esc/stuck_counter": controller_state["stuck_counter"],
            })
        
        self.experiment.logger.log(logs, step=self.experiment.n_iters_performed)