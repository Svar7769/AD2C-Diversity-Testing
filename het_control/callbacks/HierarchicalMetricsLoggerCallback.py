#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

"""
Callback to log hierarchical model metrics (estimated_snd, scaling_ratio, weights).
"""

import torch
from benchmarl.experiment.callback import Callback
from tensordict import TensorDictBase
from het_control.callbacks.callback import get_het_model
from het_control.models.het_control_mlp_hierarchical import HetControlMlpHierarchical


class HierarchicalMetricsLoggerCallback(Callback):
    """Logs hierarchical model metrics during training.
    
    Args:
        control_group (str): Name of the agent group to track (default: "agents")
        log_interval (int): Log every N training iterations (default: 100)
    """
    
    def __init__(
        self,
        control_group: str = "agents",
        log_interval: int = 100,
    ):
        super().__init__()
        self.control_group = control_group
        self.log_interval = log_interval
        self.iteration = 0
        self.model = None
    
    def on_setup(self):
        """Initialize on experiment setup."""
        try:
            policy = self.experiment.group_policies.get(self.control_group)
            if policy is None:
                print(f"⚠️  HierarchicalMetricsLogger: No policy found for group '{self.control_group}'")
                return
            
            model = get_het_model(policy)
            
            if isinstance(model, HetControlMlpHierarchical):
                self.model = model
                print(f"✅ HierarchicalMetricsLogger initialized for '{self.control_group}'")
            else:
                print(f"⚠️  HierarchicalMetricsLogger: Model is not hierarchical. Disabling logger.")
                self.model = None
        except Exception as e:
            print(f"⚠️  HierarchicalMetricsLogger initialization failed: {e}")
            self.model = None
    
    def on_train_step(
        self,
        batch: TensorDictBase,
        group: str,
    ) -> TensorDictBase:
        """Log hierarchical metrics during training."""
        
        if group != self.control_group or self.model is None:
            return batch
        
        self.iteration += 1
        if self.iteration % self.log_interval != 0:
            return batch
        
        try:
            # Log estimated SND
            estimated_snd = self.model.estimated_snd.item()
            self.experiment.logger.log(
                {f"{group}/estimated_snd": estimated_snd},
                step=self.experiment.total_frames
            )
            
            # Log desired SND for comparison
            desired_snd = self.model.desired_snd.item()
            self.experiment.logger.log(
                {f"{group}/desired_snd": desired_snd},
                step=self.experiment.total_frames
            )
            
            # Log hierarchical weights
            shared_w, subteam_w, agent_w = self.model.get_hierarchical_weights()
            
            self.experiment.logger.log(
                {
                    f"{group}/weights/shared": shared_w.item(),
                    f"{group}/weights/subteam": subteam_w.item(),
                    f"{group}/weights/agent": agent_w.item(),
                },
                step=self.experiment.total_frames
            )
            
            # Log weight sum for monitoring
            weight_sum = (shared_w + subteam_w + agent_w).item()
            self.experiment.logger.log(
                {f"{group}/weights/total": weight_sum},
                step=self.experiment.total_frames
            )
            
        except Exception as e:
            # Silently skip on error to avoid breaking training
            pass
        
        return batch