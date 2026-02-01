"""
Callback for integrating Smart Adaptive ESC with BenchMARL experiments.
"""
from typing import List, Optional, Dict, Any
import torch
import numpy as np
from tensordict import TensorDictBase
from benchmarl.experiment.callback import Callback
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpirical
from het_control.callbacks.callback import get_het_model
from het_control.callbacks.smartController import SmartAdaptiveESC


class SmartESCCallback(Callback):
    """
    Callback that uses Extremum Seeking Control to automatically tune
    the desired SND parameter during training.

    Features:
    - RMSprop adaptive learning rate
    - Vanishing perturbation based on gradient magnitude
    - Reward maximization (gradient ascent)
    """

    def __init__(
        self,
        control_group: str,
        initial_snd: float,
        min_snd: float = 0.0,
        max_snd: float = 3.0,
        # ESC parameters
        sampling_period: float = 1.0,
        dither_frequency: float = 0.5,
        dither_amplitude: float = 0.1,
        learning_rate: float = 0.01,
        high_pass_cutoff: float = 0.1,
        low_pass_cutoff: float = 0.05,
        # RMSprop parameters
        use_rmsprop: bool = True,
        beta: float = 0.9,
        epsilon: float = 1e-8,
        max_lr_multiplier: float = 10.0,
        # Vanishing perturbation
        use_vanishing_perturbation: bool = True,
        min_perturbation_ratio: float = 0.1,
    ):
        """
        Initialize the Smart ESC Callback.

        Args:
            control_group: Name of the agent group to control
            initial_snd: Initial SND (Spatial Neighbor Distance) value
            min_snd: Minimum allowed SND value
            max_snd: Maximum allowed SND value
            sampling_period: Time between ESC updates (seconds)
            dither_frequency: Perturbation frequency (rad/s)
            dither_amplitude: Initial perturbation amplitude
            learning_rate: Base learning rate for gradient ascent
            high_pass_cutoff: High-pass filter cutoff frequency (rad/s)
            low_pass_cutoff: Low-pass filter cutoff frequency (rad/s)
            use_rmsprop: Whether to use RMSprop adaptive learning rate
            beta: RMSprop decay factor (typically 0.9)
            epsilon: Small constant for numerical stability
            max_lr_multiplier: Maximum learning rate as multiple of base
            use_vanishing_perturbation: Whether to reduce perturbation as gradient decreases
            min_perturbation_ratio: Minimum perturbation as ratio of initial
        """
        super().__init__()

        # Control group configuration
        self.control_group = control_group

        # SND bounds
        self.initial_snd = initial_snd
        self.min_snd = min_snd
        self.max_snd = max_snd

        # Store ESC parameters for controller initialization and logging
        self.esc_params: Dict[str, Any] = {
            "sampling_period": sampling_period,
            "dither_frequency": dither_frequency,
            "dither_amplitude": dither_amplitude,
            "learning_rate": learning_rate,
            "high_pass_cutoff": high_pass_cutoff,
            "low_pass_cutoff": low_pass_cutoff,
            "use_rmsprop": use_rmsprop,
            "beta": beta,
            "epsilon": epsilon,
            "max_lr_multiplier": max_lr_multiplier,
            "use_vanishing_perturbation": use_vanishing_perturbation,
            "min_perturbation_ratio": min_perturbation_ratio,
        }

        # These will be initialized in on_setup
        self.model: Optional[HetControlMlpEmpirical] = None
        self.controller: Optional[SmartAdaptiveESC] = None

    def on_setup(self) -> None:
        """Initialize the controller and log hyperparameters."""
        # Log hyperparameters
        hparams = {
            "esc_control_group": self.control_group,
            "esc_initial_snd": self.initial_snd,
            "esc_min_snd": self.min_snd,
            "esc_max_snd": self.max_snd,
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
            self.controller = SmartAdaptiveESC(
                sampling_period=self.esc_params["sampling_period"],
                dither_frequency=self.esc_params["dither_frequency"],
                dither_amplitude=self.esc_params["dither_amplitude"],
                learning_rate=self.esc_params["learning_rate"],
                initial_value=self.initial_snd,
                high_pass_cutoff=self.esc_params["high_pass_cutoff"],
                low_pass_cutoff=self.esc_params["low_pass_cutoff"],
                min_output=self.min_snd,
                max_output=self.max_snd,
                use_rmsprop=self.esc_params["use_rmsprop"],
                beta=self.esc_params["beta"],
                epsilon=self.esc_params["epsilon"],
                max_lr_multiplier=self.esc_params["max_lr_multiplier"],
                use_vanishing_perturbation=self.esc_params["use_vanishing_perturbation"],
                min_perturbation_ratio=self.esc_params["min_perturbation_ratio"],
            )

            # Set initial desired SND
            self.model.desired_snd[:] = float(self.initial_snd)

            print(f"\nESC Controller initialized for group '{self.control_group}'.")
            print(f"   Initial SND: {self.initial_snd:.3f}")
            print(f"   Features:")
            print(f"     - RMSprop: {'ON' if self.esc_params['use_rmsprop'] else 'OFF'}")
            print(f"     - Vanishing perturbation: {'ON' if self.esc_params['use_vanishing_perturbation'] else 'OFF'}")
            print(f"   Parameters:")
            print(f"     - Dither: +/-{self.esc_params['dither_amplitude']:.3f} @ {self.esc_params['dither_frequency']:.2f} rad/s")
            print(f"     - Learning rate: {self.esc_params['learning_rate']:.4f} (max: {self.esc_params['learning_rate'] * self.esc_params['max_lr_multiplier']:.4f})")
            print(f"     - HPF cutoff: {self.esc_params['high_pass_cutoff']:.3f} rad/s")
            print(f"     - LPF cutoff: {self.esc_params['low_pass_cutoff']:.3f} rad/s")
            print(f"   SND bounds: [{self.min_snd:.1f}, {self.max_snd:.1f}]\n")
        else:
            print(f"\nWARNING: Compatible model not found for group '{self.control_group}'. Disabling ESC.\n")
            self.model = None

    def on_evaluation_end(self, rollouts: List[TensorDictBase]) -> None:
        """
        Update ESC controller based on evaluation episode rewards.
        
        The controller uses the mean reward as the performance metric
        to adjust the desired SND parameter via gradient ascent.
        The updated SND is immediately applied to the model for use 
        in the next training phase.
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
        
        cost = 3.2 - mean_reward  # Convert to cost for minimization

        # Store previous SND (actual value with perturbation)
        previous_snd = self.model.desired_snd.item()
        previous_setpoint = self.controller.prev_setpoint

        # Update controller with REWARD (gradient ascent)
        result = self.controller.update(cost)  # mean_reaward gradient ascent.

        # Extract results from dictionary
        output = result["output"]
        setpoint = result["setpoint"]
        gradient = result["gradient"]
        gradient_magnitude = result["gradient_magnitude"]
        adaptive_lr = result["adaptive_lr"]
        perturbation_amplitude = result["perturbation_amplitude"]
        hpf_output = result["hpf_output"]
        cost = cost

        # Clamp output to bounds
        output_clamped = np.clip(output, self.min_snd, self.max_snd)

        # ⭐ Update model immediately with perturbed output
        # This ensures training uses the new ESC-controlled SND value
        self.model.desired_snd[:] = float(output_clamped)

        # Compute actual update step
        update_step = self.model.desired_snd.item() - previous_snd
        setpoint_change = setpoint - previous_setpoint

        # Log update with detailed information
        print(
            f"[ESC] Step {self.experiment.n_iters_performed:6d} | "
            f"Reward: {mean_reward:+7.3f} ±{reward_std:5.3f} | "
            f"SND: {previous_snd:.4f} → {self.model.desired_snd.item():.4f} (Δ={update_step:+.4f}) | "
            f"Setpoint: {previous_setpoint:.4f} → {setpoint:.4f} (Δ={setpoint_change:+.4f}) | "
            f"Dither: {perturbation_amplitude:.4f} | "
            f"LR: {adaptive_lr:.6f} | "
            f"Grad: {gradient:+.5f} (RMS: {gradient_magnitude:.5f})"
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
            "esc/perturbation_current": output_clamped - setpoint,
            "esc/perturbation_raw": output - setpoint,
            "esc/perturbation_amplitude": perturbation_amplitude,

            # Controller internals
            "esc/gradient_estimate": gradient,
            "esc/gradient_magnitude": gradient_magnitude,
            "esc/hpf_output": hpf_output,
            "esc/integrator_state": self.controller.integral,
            "esc/phase": self.controller.phase,
            "esc/v": self.controller.v,
            "esc/step_count": self.controller.step_count,
        }

        self.experiment.logger.log(logs, step=self.experiment.n_iters_performed)
