
"""
Callback to log subteam assignment statistics for hierarchical models.
Tracks which agents belong to which behavioral clusters over time.
"""

from typing import Optional, List
import torch
from benchmarl.experiment import Experiment
from benchmarl.experiment.callback import Callback
from tensordict import TensorDictBase


class SubteamAssignmentLoggerCallback(Callback):
    """Logs subteam assignment statistics during training.
    
    For hierarchical models with behavioral clustering, this callback:
    - Tracks the soft/hard assignment of agents to subteams
    - Logs per-agent subteam probabilities
    - Computes subteam diversity metrics
    - Logs dominant subteam for each agent
    
    Note: Currently only logs during training, not evaluation, to avoid
    complexity with batch collection.
    
    Args:
        control_group (str): Name of the agent group to track (default: "agents")
        log_interval (int): Log every N training iterations (default: 1)
    """
    
    def __init__(
        self,
        control_group: str = "agents",
        log_interval: int = 1,
    ):
        super().__init__()
        self.control_group = control_group
        self.log_interval = log_interval
        self.iteration = 0
    
    def on_setup(self):
        """Initialize on experiment setup."""
        # Check if model has subteam assignments
        policy = self.experiment.group_policies.get(self.control_group)
        if policy is None:
            print(f"⚠️  Warning: No policy found for group '{self.control_group}'")
            return
        
        # Try to detect if this is a hierarchical model
        try:
            from het_control.callbacks.callback import get_het_model
            model = get_het_model(policy)
            has_subteams = hasattr(model, 'n_subteams')
        except (TypeError, AttributeError) as e:
            print(f"⚠️  Warning: Could not access model: {e}")
            has_subteams = False
        
        if has_subteams:
            self.n_subteams = model.n_subteams
            self.n_agents = model.n_agents
            print(f"✅ SubteamAssignmentLogger initialized: {self.n_agents} agents, {self.n_subteams} subteams")
        else:
            print(f"⚠️  Warning: Model does not have subteam assignments. Logger will be inactive.")
            self.n_subteams = None
    
    def on_train_step(
        self,
        batch: TensorDictBase,
        group: str,
    ) -> TensorDictBase:
        """Log subteam assignments during training."""
        
        # Only log for the control group
        if group != self.control_group:
            return batch
        
        # Skip if no subteams
        if self.n_subteams is None:
            return batch
        
        # Check log interval
        self.iteration += 1
        if self.iteration % self.log_interval != 0:
            return batch
        
        # Extract subteam assignments if available
        try:
            assignments = batch.get((group, "subteam_assignments"), default=None)
        except (KeyError, RuntimeError):
            # Model doesn't produce subteam assignments or key not found
            return batch
        
        if assignments is None:
            return batch
        
        # assignments shape: [batch, n_agents, n_subteams]
        # Average over batch dimension
        avg_assignments = assignments.mean(dim=0)  # [n_agents, n_subteams]
        
        # Log per-agent subteam probabilities
        for agent_idx in range(self.n_agents):
            for subteam_idx in range(self.n_subteams):
                prob = avg_assignments[agent_idx, subteam_idx].item()
                self.experiment.logger.log(
                    {f"{group}/agent_{agent_idx}/subteam_{subteam_idx}_prob": prob},
                    step=self.experiment.total_frames
                )
        
        # Log dominant subteam for each agent (argmax)
        dominant_subteams = torch.argmax(avg_assignments, dim=-1)  # [n_agents]
        for agent_idx in range(self.n_agents):
            dominant = dominant_subteams[agent_idx].item()
            self.experiment.logger.log(
                {f"{group}/agent_{agent_idx}/dominant_subteam": dominant},
                step=self.experiment.total_frames
            )
        
        # Compute and log subteam diversity metrics
        # Entropy of assignment distribution (higher = more uncertain/diverse)
        # H = -sum(p * log(p))
        assignment_entropy = -(avg_assignments * torch.log(avg_assignments + 1e-10)).sum(dim=-1)  # [n_agents]
        avg_entropy = assignment_entropy.mean().item()
        
        self.experiment.logger.log(
            {f"{group}/subteam_assignment_entropy": avg_entropy},
            step=self.experiment.total_frames
        )
        
        # Compute subteam population (how many agents prefer each subteam)
        subteam_populations = torch.zeros(self.n_subteams, device=assignments.device)
        for subteam_idx in range(self.n_subteams):
            # Count agents whose dominant subteam is this one
            count = (dominant_subteams == subteam_idx).sum().item()
            subteam_populations[subteam_idx] = count
            self.experiment.logger.log(
                {f"{group}/subteam_{subteam_idx}_population": count},
                step=self.experiment.total_frames
            )
        
        # Compute subteam balance (how evenly distributed agents are)
        # Perfect balance = 1.0, complete imbalance = 0.0
        if self.n_agents > 0:
            expected_per_subteam = self.n_agents / self.n_subteams
            imbalance = ((subteam_populations - expected_per_subteam) ** 2).sum() / self.n_agents
            balance = 1.0 / (1.0 + imbalance)
            self.experiment.logger.log(
                {f"{group}/subteam_balance": balance},
                step=self.experiment.total_frames
            )
        
        return batch