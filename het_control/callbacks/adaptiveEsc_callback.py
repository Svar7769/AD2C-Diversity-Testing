"""
Callback for integrating ESC with ML optimizers into BenchMARL experiments.
"""
from typing import List, Optional, Literal, Tuple
import torch
import numpy as np
from tensordict import TensorDictBase
from benchmarl.experiment.callback import Callback
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpirical
from het_control.callbacks.callback import get_het_model
from het_control.callbacks.adaptiveEsc import ExtremumSeekingController


class AdaptiveESCCallback(Callback):
    """
    ESC callback using ML-style optimizers (Adam, RMSprop, SGD).
    
    Automatically tunes the desired SND parameter during training
    based on episode rewards using gradient estimates from ESC.
    """
    
    def __init__(
        self,
        control_group: str,
        initial_snd: float,
        dither_magnitude: float = 0.1,
        dither_frequency_rad_s: float = 0.5,
        high_pass_cutoff_rad_s: float = 0.1,
        low_pass_cutoff_rad_s: float = 0.05,
        sampling_period: float = 1.0,
        min_snd: float = 0.0,
        max_snd: float = 3.0,
        maximize: bool = True,
        # Optimizer configuration
        optimizer_type: Literal['adam', 'rmsprop', 'sgd'] = 'adam',
        learning_rate: float = 0.01,
        # Optimizer-specific hyperparameters
        momentum: float = 0.9,
        betas: Tuple[float, float] = (0.9, 0.999),
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        centered: bool = False,
        amsgrad: bool = False,
        nesterov: bool = False,
        # Learning rate scheduler (not implemented yet)
        use_lr_scheduler: bool = False,
        scheduler_type: Literal['step', 'exponential', 'cosine', 'plateau'] = 'plateau',
        scheduler_patience: int = 100,
        scheduler_factor: float = 0.5
    ):
        super().__init__()
        self.control_group = control_group
        self.initial_snd = initial_snd
        self.min_snd = min_snd
        self.max_snd = max_snd
        self.maximize = maximize
        
        # Store all parameters
        self.esc_params = {
            "sampling_period": sampling_period,
            "dither_frequency": dither_frequency_rad_s,
            "dither_magnitude": dither_magnitude,
            "initial_snd": initial_snd,
            "high_pass_cutoff": high_pass_cutoff_rad_s,
            "low_pass_cutoff": low_pass_cutoff_rad_s,
            "min_snd": min_snd,
            "max_snd": max_snd,
            "maximize": maximize,
            "optimizer_type": optimizer_type,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "betas": betas,
            "alpha": alpha,
            "eps": eps,
            "weight_decay": weight_decay,
            "centered": centered,
            "amsgrad": amsgrad,
            "nesterov": nesterov,
        }
        
        # Optimizer kwargs (filter out None and unused params)
        self.optimizer_kwargs = {}
        if optimizer_type == 'adam':
            self.optimizer_kwargs = {
                "beta1": betas[0],
                "beta2": betas[1],
                "eps": eps,
                "amsgrad": amsgrad
            }
        elif optimizer_type == 'rmsprop':
            self.optimizer_kwargs = {
                "alpha": alpha,
                "eps": eps,
                "momentum": momentum,
                "centered": centered
            }
        elif optimizer_type == 'sgd':
            self.optimizer_kwargs = {
                "momentum": momentum,
                "nesterov": nesterov
            }
        
        self.model: Optional[HetControlMlpEmpirical] = None
        self.controller: Optional[ExtremumSeekingController] = None

    def on_setup(self) -> None:
        """Initialize the controller and log hyperparameters."""
        # Log hyperparameters
        hparams = {
            "esc_control_group": self.control_group,
            **{f"esc_{k}": v for k, v in self.esc_params.items() if v is not None}
        }
        self.experiment.logger.log_hparams(**hparams)
        
        # Verify control group exists
        if self.control_group not in self.experiment.group_policies:
            print(f"\n⚠️  WARNING: ESC control group '{self.control_group}' not found. Disabling controller.\n")
            return
        
        # Get model
        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)
        
        # Initialize controller if model is compatible
        if not isinstance(self.model, HetControlMlpEmpirical):
            print(f"\n⚠️  WARNING: Compatible model not found for group '{self.control_group}'. Disabling ESC.\n")
            self.model = None
            return
        
        # Create controller
        self.controller = ExtremumSeekingController(
            sampling_period=self.esc_params["sampling_period"],
            dither_frequency=self.esc_params["dither_frequency"],
            dither_magnitude=self.esc_params["dither_magnitude"],
            initial_value=self.initial_snd,
            high_pass_cutoff=self.esc_params["high_pass_cutoff"],
            low_pass_cutoff=self.esc_params["low_pass_cutoff"],
            min_output=self.min_snd,
            maximize=self.maximize,
            optimizer_type=self.esc_params["optimizer_type"],
            learning_rate=self.esc_params["learning_rate"],
            **self.optimizer_kwargs
        )
        
        # Set initial desired SND (with perturbation)
        initial_perturbation = self.controller.a * np.sin(self.controller.phase)
        initial_output = np.clip(
            self.controller.theta + initial_perturbation,
            self.min_snd,
            self.max_snd
        )
        self.model.desired_snd[:] = float(initial_output)
        
        # Print setup summary
        self._print_setup_summary()
    
    def _print_setup_summary(self):
        """Print controller setup summary."""
        print(f"\n✅ ESC Controller initialized for group '{self.control_group}'")
        print(f"   Initial SND: {self.initial_snd:.3f}")
        print(f"   Objective: {'MAXIMIZE reward' if self.maximize else 'MINIMIZE cost'}")
        print(f"   Optimizer: {self.esc_params['optimizer_type'].upper()}")
        print(f"   Learning rate: {self.esc_params['learning_rate']:.4f}")
        
        # Optimizer-specific details
        if self.esc_params["optimizer_type"] == 'adam':
            print(f"   Betas: {self.esc_params['betas']}")
            print(f"   AMSGrad: {self.esc_params['amsgrad']}")
        elif self.esc_params["optimizer_type"] == 'rmsprop':
            print(f"   Alpha: {self.esc_params['alpha']}")
            print(f"   Momentum: {self.esc_params['momentum']}")
            print(f"   Centered: {self.esc_params['centered']}")
        elif self.esc_params["optimizer_type"] == 'sgd':
            print(f"   Momentum: {self.esc_params['momentum']}")
            print(f"   Nesterov: {self.esc_params['nesterov']}")
        
        print(f"   Dither: ±{self.esc_params['dither_magnitude']:.3f} @ {self.esc_params['dither_frequency']:.2f} rad/s")
        print(f"   Filter cutoffs: HPF={self.esc_params['high_pass_cutoff']:.3f}, LPF={self.esc_params['low_pass_cutoff']:.3f} rad/s")
        print(f"   SND bounds: [{self.min_snd:.1f}, {self.max_snd:.1f}]\n")

    def on_evaluation_end(self, rollouts: List[TensorDictBase]) -> None:
        """Update ESC controller based on evaluation rewards."""
        if self.model is None or self.controller is None:
            return
        
        # Collect episode rewards
        episode_rewards = []
        with torch.no_grad():
            for rollout in rollouts:
                reward_key = ('next', self.control_group, 'reward')
                if reward_key in rollout.keys(include_nested=True):
                    total_reward = rollout.get(reward_key).sum().item()
                    episode_rewards.append(total_reward)
        
        if not episode_rewards:
            print("\n⚠️  WARNING: No episode rewards found. Skipping ESC update.\n")
            return
        
        # Compute statistics
        mean_reward = np.mean(episode_rewards)
        reward_std = np.std(episode_rewards)
        
        # Prepare metric for controller
        metric = mean_reward if self.maximize else -mean_reward
        
        # Store previous values
        previous_snd = self.model.desired_snd.item()
        previous_theta = self.controller.theta
        
        # Update controller
        (
            perturbed_output,
            hpf_output,
            gradient_estimate,
            gradient_magnitude,
            _,
            theta_value
        ) = self.controller.update(metric)
        
        # Clamp and update model
        perturbed_output_clamped = np.clip(perturbed_output, self.min_snd, self.max_snd)
        self.model.desired_snd[:] = float(perturbed_output_clamped)
        
        # Compute changes
        snd_change = self.model.desired_snd.item() - previous_snd
        theta_change = theta_value - previous_theta
        
        # Get optimizer state
        controller_state = self.controller.get_state()
        current_lr = controller_state.get("learning_rate", self.esc_params["learning_rate"])
        
        # Build optimizer info string
        optimizer_info = f"LR={current_lr:.6f}"
        if self.esc_params["optimizer_type"] == 'adam':
            if "momentum" in controller_state:
                optimizer_info += f" | m={controller_state['momentum']:.4f}"
            if "variance" in controller_state:
                optimizer_info += f" | v={controller_state['variance']:.4f}"
        
        # Print update
        print(
            f"[ESC-{self.esc_params['optimizer_type'].upper()}] "
            f"Step {self.experiment.n_iters_performed:6d} | "
            f"Reward: {mean_reward:+7.3f} ±{reward_std:5.3f} | "
            f"SND: {previous_snd:.4f} → {self.model.desired_snd.item():.4f} (Δ={snd_change:+.4f}) | "
            f"θ: {previous_theta:.4f} → {theta_value:.4f} (Δ={theta_change:+.4f}) | "
            f"∇: {gradient_estimate:+.5f} (||∇||={gradient_magnitude:.5f}) | "
            f"{optimizer_info}"
        )
        
        # Log metrics
        logs = {
            "esc/reward_mean": mean_reward,
            "esc/reward_std": reward_std,
            "esc/metric": metric,
            "esc/snd_actual": self.model.desired_snd.item(),
            "esc/snd_change": snd_change,
            "esc/theta": theta_value,
            "esc/theta_change": theta_change,
            "esc/perturbation": perturbed_output_clamped - theta_value,
            "esc/gradient_estimate": gradient_estimate,
            "esc/gradient_magnitude": gradient_magnitude,
            "esc/hpf_output": hpf_output,
            "esc/phase": self.controller.phase,
            "esc/learning_rate": current_lr,
            "esc/step_count": controller_state.get("step_count", 0),
        }
        
        # Add optimizer-specific logs
        if self.esc_params["optimizer_type"] == 'adam':
            if "momentum" in controller_state:
                logs["esc/adam_m"] = controller_state["momentum"]
            if "variance" in controller_state:
                logs["esc/adam_v"] = controller_state["variance"]
        elif self.esc_params["optimizer_type"] == 'rmsprop':
            if "v" in controller_state:
                logs["esc/rmsprop_v"] = controller_state["v"]
            if "momentum_buffer" in controller_state:
                logs["esc/rmsprop_buf"] = controller_state["momentum_buffer"]
        elif self.esc_params["optimizer_type"] == 'sgd':
            if "momentum_buffer" in controller_state:
                logs["esc/sgd_momentum_buf"] = controller_state["momentum_buffer"]
        
        self.experiment.logger.log(logs, step=self.experiment.n_iters_performed)