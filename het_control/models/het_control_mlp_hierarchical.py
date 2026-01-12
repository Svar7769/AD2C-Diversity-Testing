#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Type, Sequence, Optional

import torch
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor
from torch import nn
from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
from het_control.callbacks.snd import compute_behavioral_distance
from het_control.callbacks.utils import overflowing_logits_norm
from .utils import squash


class HetControlMlpHierarchical(Model):
    """Three-part hierarchical heterogeneous control model.
    
    Architecture:
    - Shared MLP: Team-level baseline policy
    - Subteam MLPs: K behavioral clusters learned via soft assignment
    - Agent MLPs: Individual agent specializations
    
    Final output = shared_weight * shared + subteam_weight * subteam[assignment] + agent_weight * agent
    """
    
    def __init__(
        self,
        activation_class: Type[nn.Module],
        num_cells: Sequence[int],
        n_subteams: int,
        desired_snd: float,
        probabilistic: bool,
        scale_mapping: Optional[str],
        tau: float,
        subteam_tau: float,
        bootstrap_from_desired_snd: bool,
        process_shared: bool,
        shared_weight_init: float,
        subteam_weight_init: float,
        agent_weight_init: float,
        use_hard_assignment: bool,
        **kwargs,
    ):
        """Three-part hierarchical DiCo policy model.
        
        Args:
            activation_class (Type[nn.Module]): activation class to be used.
            num_cells (int or Sequence[int], optional): number of cells of every layer in between the input and output.
            n_subteams (int): Number of behavioral subteam clusters (typically 2-4).
            desired_snd (float): The desired SND diversity.
            probabilistic (bool): Whether the model has stochastic actions or not.
            scale_mapping (str, optional): Type of mapping to use to make the std_dev output of the policy positive
                (choices: "softplus", "exp", "relu", "biased_softplus_1")
            tau (float): The soft-update parameter of the estimated diversity. Must be between 0 and 1.
            subteam_tau (float): Temperature for soft subteam assignment. Lower values make assignments harder.
            bootstrap_from_desired_snd (bool): Whether on the first iteration the estimated SND should be bootstrapped
                from the desired snd (True) or from the measured SND (False).
            process_shared (bool): Whether to process the homogeneous part of the policy with a tanh squashing operation.
            shared_weight_init (float): Initial weight for shared component contribution.
            subteam_weight_init (float): Initial weight for subteam component contribution.
            agent_weight_init (float): Initial weight for agent-specific component contribution.
            use_hard_assignment (bool): If True, use hard (argmax) subteam assignment instead of soft (softmax).
        """
        super().__init__(**kwargs)

        self.num_cells = num_cells
        self.activation_class = activation_class
        self.probabilistic = probabilistic
        self.scale_mapping = scale_mapping
        self.tau = tau
        self.subteam_tau = subteam_tau
        self.bootstrap_from_desired_snd = bootstrap_from_desired_snd
        self.process_shared = process_shared
        self.n_subteams = n_subteams
        self.use_hard_assignment = use_hard_assignment

        # Diversity tracking buffers
        self.register_buffer(
            name="desired_snd",
            tensor=torch.tensor([desired_snd], device=self.device, dtype=torch.float),
        )
        self.register_buffer(
            name="estimated_snd",
            tensor=torch.tensor([float("nan")], device=self.device, dtype=torch.float),
        )

        # Scale extractor for probabilistic policies
        self.scale_extractor = (
            NormalParamExtractor(scale_mapping=scale_mapping)
            if scale_mapping is not None
            else None
        )

        self.input_features = self.input_leaf_spec.shape[-1]
        self.output_features = self.output_leaf_spec.shape[-1]

        # 1. SHARED MLP (Team-level baseline)
        self.shared_mlp = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=False,
            share_params=True,  # Parameter-shared across all agents
            device=self.device,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
        )

        # 2. SUBTEAM MLPs (Behavioral clusters)
        agent_outputs = (
            self.output_features // 2 if self.probabilistic else self.output_features
        )
        
        # Create K subteam networks (parameter-shared within each subteam)
        self.subteam_mlps = nn.ModuleList([
            MultiAgentMLP(
                n_agent_inputs=self.input_features,
                n_agent_outputs=agent_outputs,
                n_agents=self.n_agents,
                centralised=False,
                share_params=True,  # Shared within this subteam
                device=self.device,
                activation_class=self.activation_class,
                num_cells=self.num_cells,
            )
            for _ in range(self.n_subteams)
        ])
        
        # Subteam assignment network (learns behavioral clustering)
        self.assignment_network = nn.Sequential(
            nn.Linear(self.input_features, self.num_cells[0], device=self.device),
            self.activation_class(),
            nn.Linear(self.num_cells[0], self.n_subteams, device=self.device),
        )

        # 3. AGENT-SPECIFIC MLPs (Individual specialization)
        self.agent_mlps = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=agent_outputs,
            n_agents=self.n_agents,
            centralised=False,
            share_params=False,  # NOT parameter-shared - agent-specific
            device=self.device,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
        )
        
        # Learnable hierarchical composition weights
        self.register_parameter(
            "shared_weight",
            nn.Parameter(torch.tensor(shared_weight_init, device=self.device))
        )
        self.register_parameter(
            "subteam_weight", 
            nn.Parameter(torch.tensor(subteam_weight_init, device=self.device))
        )
        self.register_parameter(
            "agent_weight",
            nn.Parameter(torch.tensor(agent_weight_init, device=self.device))
        )

    def _perform_checks(self):
        """Perform BenchMARL-specific validation checks."""
        super()._perform_checks()

        if self.centralised or not self.input_has_agent_dim:
            raise ValueError(f"{self.__class__.__name__} can only be used for policies")

        if self.input_has_agent_dim and self.input_leaf_spec.shape[-2] != self.n_agents:
            raise ValueError(
                "If the MLP input has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the MLP output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )
        
        if self.n_subteams < 1:
            raise ValueError(f"n_subteams must be >= 1, got {self.n_subteams}")
        
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau must be in [0, 1], got {self.tau}")
        
        if self.subteam_tau <= 0:
            raise ValueError(f"subteam_tau must be positive, got {self.subteam_tau}")

    def compute_subteam_assignment(
        self, 
        input: torch.Tensor,
    ) -> torch.Tensor:
        """Compute soft (or hard) assignment to subteams.
        
        Args:
            input: [*batch, n_agents, n_features]
            
        Returns:
            assignment_weights: [*batch, n_agents, n_subteams]
        """
        # Compute logits for each agent's subteam assignment
        assignment_logits = self.assignment_network(input)  # [*batch, n_agents, n_subteams]
        
        if self.use_hard_assignment:
            # Hard assignment (one-hot) - useful for interpretability
            assignment_idx = torch.argmax(assignment_logits, dim=-1)
            assignment_weights = torch.nn.functional.one_hot(
                assignment_idx, num_classes=self.n_subteams
            ).float()
        else:
            # Soft assignment with temperature - allows gradient flow
            assignment_weights = torch.softmax(
                assignment_logits / self.subteam_tau, dim=-1
            )
        
        return assignment_weights

    def _forward(
        self,
        tensordict: TensorDictBase,
        agent_index: int = None,
        update_estimate: bool = True,
        compute_estimate: bool = True,
    ) -> TensorDictBase:
        """Forward pass with three-part hierarchical composition."""
        
        # Gather input
        input = tensordict.get(self.in_key)  # [*batch, n_agents, n_features]
        
        # ========== 1. SHARED OUTPUT (Team baseline) ==========
        shared_out = self.shared_mlp.forward(input)
        shared_out = self.process_shared_out(shared_out)
        
        # ========== 2. SUBTEAM OUTPUT (Behavioral clusters) ==========
        # Compute soft/hard assignments to subteams
        subteam_assignments = self.compute_subteam_assignment(input)  # [*batch, n_agents, n_subteams]
        
        # Get outputs from all subteam networks
        subteam_outputs = []
        for subteam_mlp in self.subteam_mlps:
            subteam_out = subteam_mlp.forward(input)
            if self.probabilistic:
                # Only use mean component from subteams (variance comes from shared)
                subteam_out, _ = subteam_out.chunk(2, -1)
            subteam_outputs.append(subteam_out)
        
        # Stack and compute weighted combination
        subteam_stack = torch.stack(subteam_outputs, dim=-2)  # [*batch, n_agents, n_subteams, action_dim]
        # Weighted sum over subteams: each agent's output is a mixture of subteam outputs
        subteam_out = torch.einsum(
            '...nsa,...ns->...na',
            subteam_stack,
            subteam_assignments
        )  # [*batch, n_agents, action_dim]
        
        # ========== 3. AGENT-SPECIFIC OUTPUT (Individual specialization) ==========
        if agent_index is None:
            # Gather outputs for all agents
            agent_out = self.agent_mlps.forward(input)
        else:
            # Gather output for specific agent (used during diversity estimation)
            agent_out = self.agent_mlps.agent_networks[agent_index].forward(input)
        
        if self.probabilistic:
            # Only use mean component (variance comes from shared)
            agent_out, _ = agent_out.chunk(2, -1)
        
        # ========== DIVERSITY SCALING (DiCo mechanism) ==========
        if (
            self.desired_snd > 0
            and torch.is_grad_enabled()  # we are training
            and compute_estimate
            and self.n_agents > 1
        ):
            # Update \widehat{SND}
            distance = self.estimate_snd(input)
            if update_estimate:
                self.estimated_snd[:] = distance.detach()
        else:
            distance = self.estimated_snd
            
        if self.desired_snd == 0:
            scaling_ratio = 0.0
        elif (
            self.desired_snd == -1  # Unconstrained networks
            or distance.isnan().any()  # First iteration
            or self.n_agents == 1
        ):
            scaling_ratio = 1.0
        else:  # DiCo scaling
            scaling_ratio = torch.where(
                distance != self.desired_snd,
                self.desired_snd / distance,
                torch.ones_like(distance),
            )
        
        # ========== HIERARCHICAL COMPOSITION ==========
        if self.probabilistic:
            shared_loc, shared_scale = shared_out.chunk(2, -1)
            
            # Combine: weighted_shared + weighted_subteam + weighted_agent
            # Only subteam and agent components are scaled by diversity
            agent_loc = (
                self.shared_weight * shared_loc +
                self.subteam_weight * subteam_out * scaling_ratio +
                self.agent_weight * agent_out * scaling_ratio
            )
            
            # Compute overflow norm for logging
            out_loc_norm = overflowing_logits_norm(
                agent_loc, self.action_spec[self.agent_group, "action"]
            )
            
            # Use shared scale for all components
            agent_scale = shared_scale
            out = torch.cat([agent_loc, agent_scale], dim=-1)
        else:
            # Deterministic case
            out = (
                self.shared_weight * shared_out +
                self.subteam_weight * subteam_out * scaling_ratio +
                self.agent_weight * agent_out * scaling_ratio
            )
            out_loc_norm = overflowing_logits_norm(
                out, self.action_spec[self.agent_group, "action"]
            )
        
        # ========== STORE DIAGNOSTICS IN TENSORDICT ==========
        tensordict.set(
            (self.agent_group, "subteam_assignments"),
            subteam_assignments,
        )
        tensordict.set(
            (self.agent_group, "estimated_snd"),
            self.estimated_snd.expand(tensordict.get_item_shape(self.agent_group)),
        )
        tensordict.set(
            (self.agent_group, "scaling_ratio"),
            (
                torch.tensor(scaling_ratio, device=self.device).expand_as(out)
                if not isinstance(scaling_ratio, torch.Tensor)
                else scaling_ratio.expand_as(out)
            ),
        )
        tensordict.set((self.agent_group, "logits"), out)
        tensordict.set((self.agent_group, "out_loc_norm"), out_loc_norm)
        
        # Store hierarchical weights for monitoring
        tensordict.set(
            (self.agent_group, "shared_weight"),
            self.shared_weight.expand(tensordict.get_item_shape(self.agent_group)),
        )
        tensordict.set(
            (self.agent_group, "subteam_weight"),
            self.subteam_weight.expand(tensordict.get_item_shape(self.agent_group)),
        )
        tensordict.set(
            (self.agent_group, "agent_weight"),
            self.agent_weight.expand(tensordict.get_item_shape(self.agent_group)),
        )

        tensordict.set(self.out_key, out)
        return tensordict

    def process_shared_out(self, logits: torch.Tensor):
        """Process shared output (same as original HetControlMlpEmpirical)."""
        if not self.probabilistic and self.process_shared:
            return squash(
                logits,
                action_spec=self.action_spec[self.agent_group, "action"],
                clamp=False,
            )
        elif self.probabilistic:
            loc, scale = self.scale_extractor(logits)
            if self.process_shared:
                loc = squash(
                    loc,
                    action_spec=self.action_spec[self.agent_group, "action"],
                    clamp=False,
                )
            return torch.cat([loc, scale], dim=-1)
        else:
            return logits

    def estimate_snd(self, obs: torch.Tensor):
        """Update \widehat{SND} using behavioral distance between agent policies."""
        agent_actions = []
        # Gather what actions each agent would take given the obs tensor
        for agent_net in self.agent_mlps.agent_networks:
            agent_outputs = agent_net(obs)
            agent_actions.append(agent_outputs)

        # Compute the SND of these unscaled agent-specific policies
        distance = (
            compute_behavioral_distance(agent_actions=agent_actions, just_mean=True)
            .mean()
            .unsqueeze(-1)
        )
        
        if self.estimated_snd.isnan().any():  # First iteration
            distance = self.desired_snd if self.bootstrap_from_desired_snd else distance
        else:
            # Soft update of \widehat{SND}
            distance = (1 - self.tau) * self.estimated_snd + self.tau * distance

        return distance


@dataclass
class HetControlMlpHierarchicalConfig(ModelConfig):
    """Configuration for three-part hierarchical heterogeneous control model."""
    
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING
    
    # Subteam clustering parameters
    n_subteams: int = 3
    subteam_tau: float = 0.1
    use_hard_assignment: bool = False
    
    # Diversity parameters
    desired_snd: float = MISSING
    tau: float = MISSING
    bootstrap_from_desired_snd: bool = MISSING
    
    # Architecture parameters
    process_shared: bool = MISSING
    probabilistic: Optional[bool] = MISSING
    scale_mapping: Optional[str] = MISSING
    
    # Hierarchical composition weights
    shared_weight_init: float = 1.0
    subteam_weight_init: float = 0.5
    agent_weight_init: float = 0.25

    @staticmethod
    def associated_class():
        return HetControlMlpHierarchical