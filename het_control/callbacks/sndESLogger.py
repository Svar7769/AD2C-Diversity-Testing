from typing import Optional
import torch
from tensordict import TensorDictBase
from benchmarl.experiment import Experiment
from benchmarl.experiment.callback import Callback
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpirical
from het_control.callbacks.callback import get_het_model


class TrajectorySNDLoggerCallback(Callback):
    """Logs ESC controller values during training."""
    
    def __init__(self, control_group: str):
        super().__init__()
        self.control_group = control_group
        self.collect_step_count = 0
        self.model: Optional[HetControlMlpEmpirical] = None

    def on_setup(self) -> None:
        """Initialize and validate the model."""
        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Logger group '{self.control_group}' not found. Disabling logger.\n")
            return
        
        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)
        
        if isinstance(self.model, HetControlMlpEmpirical):
            print(f"\nSUCCESS: Logger initialized for HetControlMlpEmpirical on group '{self.control_group}'.")
        else:
            print(f"\nWARNING: Compatible HetControlMlpEmpirical model not found. Disabling logger.\n")
            self.model = None

    def on_batch_collected(self, batch: TensorDictBase) -> None:
        """Process batch data and log ESC controller values."""
        if not isinstance(self.model, HetControlMlpEmpirical):
            return
        
        self._log_esc_scalars(self.experiment, batch)
        self.collect_step_count += 1

    def _log_esc_scalars(self, experiment: Experiment, batch: TensorDictBase) -> None:
        """Log scalar values from the batch."""
        to_log = {}

        # Top-level keys in the agent group
        top_level_keys = ["estimated_snd", "scaling_ratio", "current_dither", "target_diversity", "k_hat"]
        for key in top_level_keys:
            val = batch.get((self.control_group, key), None)
            if val is not None:
                to_log[f"collection/{self.control_group}/{key}"] = val.float().mean().item()

        # Keys inside the 'esc_learning' namespace
        if (self.control_group, "esc_learning") in batch.keys(include_nested=True):
            esc_learning_keys = [
                "reward_mean", "hpf_out", "lpf_out", "gradient_final",
                "k_hat", "integral", "m2_sqrt", "wt"
            ]
            for key in esc_learning_keys:
                val = batch.get((self.control_group, "esc_learning", key), None)
                if val is not None:
                    # Log the post-update k_hat under a distinct name to avoid confusion
                    log_key = "target_diversity" if key == "k_hat" else key
                    to_log[f"controller_learning/{self.control_group}/{log_key}"] = val.float().mean().item()

        if to_log:
            experiment.logger.log(to_log, step=self.collect_step_count)
