
from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Type, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor
from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
from het_control.callbacks.snd import compute_behavioral_distance
from .utils import squash


class HetControlMlpHierarchical(Model):
    """
    Hierarchical heterogeneous control policy:
      - Shared team baseline
      - K subteam (role) policies
      - Agent-specific specialists
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
        subteam_weight_init: float,
        agent_weight_init: float,
        use_hard_assignment: bool,
        selective_subteam_sharing: bool = False,
        normalize_weights: bool = False,
        clip_weights: bool = True,
        use_layer_norm: bool = True,
        **kwargs,
    ):
        # Required for BenchMARL checks
        self.num_cells = num_cells
        self.activation_class = activation_class
        self.probabilistic = probabilistic
        self.n_subteams = n_subteams
        self.tau = tau
        self.subteam_tau = subteam_tau
        self.bootstrap_from_desired_snd = bootstrap_from_desired_snd
        self.process_shared = process_shared
        self.use_hard_assignment = use_hard_assignment
        self.selective_subteam_sharing = selective_subteam_sharing
        self.normalize_weights = normalize_weights
        self.clip_weights = clip_weights

        super().__init__(**kwargs)

        # --------------------------------------------------
        # Buffers
        # --------------------------------------------------
        self.register_buffer("desired_snd", torch.tensor([desired_snd], device=self.device))
        self.register_buffer("estimated_snd", torch.tensor([float("nan")], device=self.device))

        self.scale_extractor = (
            NormalParamExtractor(scale_mapping=scale_mapping)
            if scale_mapping is not None
            else None
        )

        self.input_features = self.input_leaf_spec.shape[-1]
        self.output_features = self.output_leaf_spec.shape[-1]
        self.act_dim = self.output_features // 2 if probabilistic else self.output_features

        # --------------------------------------------------
        # Input normalization
        # --------------------------------------------------
        self.input_norm = (
            nn.LayerNorm(self.input_features, device=self.device)
            if use_layer_norm
            else nn.Identity()
        )

        # --------------------------------------------------
        # Shared baseline (team policy)
        # --------------------------------------------------
        self.shared_mlp = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=False,
            share_params=True,
            device=self.device,
            activation_class=activation_class,
            num_cells=num_cells,
        )

        # --------------------------------------------------
        # Agent-specific specialists
        # --------------------------------------------------
        self.agent_mlps = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=self.act_dim,
            n_agents=self.n_agents,
            centralised=False,
            share_params=False,
            device=self.device,
            activation_class=activation_class,
            num_cells=num_cells,
        )

        # --------------------------------------------------
        # Subteam (role) policies
        # --------------------------------------------------
        self.subteam_mlps = nn.ModuleList([
            MultiAgentMLP(
                n_agent_inputs=self.input_features,
                n_agent_outputs=self.act_dim,
                n_agents=self.n_agents,
                centralised=False,
                share_params=True,
                device=self.device,
                activation_class=activation_class,
                num_cells=num_cells,
            )
            for _ in range(n_subteams)
        ])
        

        # --------------------------------------------------
        # Routing network (per-agent)
        # --------------------------------------------------
        self.assignment_net = nn.Sequential(
            nn.Linear(self.input_features, num_cells[0], device=self.device),
            activation_class(),
            nn.Linear(num_cells[0], n_subteams, device=self.device),
        )

        # --------------------------------------------------
        # Learnable composition weights
        # --------------------------------------------------
        self.w_subteam = nn.Parameter(torch.tensor(subteam_weight_init, device=self.device))
        self.w_agent = nn.Parameter(torch.tensor(agent_weight_init, device=self.device))

    # ======================================================
    # Forward
    # ======================================================
    def _forward(
        self,
        tensordict: TensorDictBase,
        agent_index: int = None,
        update_estimate: bool = True,
        compute_estimate: bool = True,
    ) -> TensorDictBase:

        obs = self.input_norm(tensordict.get(self.in_key))  # [*, N, obs_dim]

        # ---------------- Shared ----------------
        shared_out = self.shared_mlp(obs)

        if self.probabilistic:
            mu_shared, sigma_shared = self.scale_extractor(shared_out)
        else:
            mu_shared, sigma_shared = shared_out, None

        if self.process_shared:
            mu_shared = squash(
                mu_shared,
                self.action_spec[self.agent_group, "action"],
                clamp=False,
            )

        # ---------------- Agent specialists ----------------
        mu_agent = (
            self.agent_mlps(obs)
            if agent_index is None
            else self.agent_mlps.agent_networks[agent_index](obs)
        )

        if agent_index is not None:
            mu_agent = mu_agent[..., agent_index, :]

        # ---------------- Subteam routing ----------------
        logits = self.assignment_net(obs)

        if self.selective_subteam_sharing:
            assign_w = F.one_hot(logits.argmax(-1), self.n_subteams).float()
        else:
            if self.training and self.use_hard_assignment:
                assign_w = F.gumbel_softmax(logits, tau=self.subteam_tau, hard=True)
            elif self.use_hard_assignment:
                assign_w = F.one_hot(logits.argmax(-1), self.n_subteams).float()
            else:
                assign_w = torch.softmax(logits / self.subteam_tau, dim=-1)

        sub_outs = torch.stack([mlp(obs) for mlp in self.subteam_mlps], dim=-2)
        mu_subteam = torch.einsum("...nka,...nk->...na", sub_outs, assign_w)

        if agent_index is not None:
            mu_subteam = mu_subteam[..., agent_index, :]

        # ---------------- SND scaling ----------------
        if agent_index is None:
            scaling = self._get_snd_scaling(mu_agent, update_estimate, compute_estimate)
        else:
            scaling = 1.0

        wk, wa = self.hierarchical_weights

        if agent_index is not None:
            mu_shared = mu_shared[..., agent_index, :]
            if sigma_shared is not None:
                sigma_shared = sigma_shared[..., agent_index, :]

        # ---------------- Final composition ----------------
        deviation = wk * mu_subteam + wa * mu_agent
        mu_final = mu_shared + scaling * deviation

        if self.probabilistic:
            out = torch.cat([mu_final, sigma_shared], dim=-1)
        else:
            out = mu_final

        # ✅ CRITICAL: write in-place (NO clone)
        tensordict.set(self.out_key, out)
        return tensordict

    # ======================================================
    # Utilities
    # ======================================================
    @property
    def hierarchical_weights(self):
        wk, wa = self.w_subteam, self.w_agent
        if self.clip_weights:
            wk, wa = wk.clamp(min=0), wa.clamp(min=0)
        if self.normalize_weights:
            s = wk + wa + 1e-8
            return wk / s, wa / s
        return wk, wa

    def get_hierarchical_weights(self):
        """Compatibility helper for callbacks expecting explicit weight access."""
        return self.hierarchical_weights

    def compute_subteam_assignment(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute per-agent subteam assignment weights from observations.
        Uses deterministic routing for logging.
        """
        normalized_obs = self.input_norm(obs)
        logits = self.assignment_net(normalized_obs)
        if self.selective_subteam_sharing or self.use_hard_assignment:
            return F.one_hot(logits.argmax(-1), self.n_subteams).float()
        return torch.softmax(logits / self.subteam_tau, dim=-1)

    def _get_snd_scaling(self, agent_means, update, compute):
        if (
            self.desired_snd > 0
            and torch.is_grad_enabled()
            and compute
            and agent_means is not None
            and self.n_agents > 1
        ):
            acts = list(agent_means.unbind(dim=-2))
            snd = compute_behavioral_distance(acts, just_mean=True).mean().unsqueeze(-1)

            if update:
                if self.estimated_snd.isnan().any():
                    self.estimated_snd[:] = (
                        self.desired_snd if self.bootstrap_from_desired_snd else snd.detach()
                    )
                else:
                    self.estimated_snd[:] = (
                        (1 - self.tau) * self.estimated_snd + self.tau * snd.detach()
                    )

        if self.desired_snd <= 0 or self.estimated_snd.isnan().any():
            return 1.0

        return self.desired_snd / self.estimated_snd

    def _perform_checks(self):
        super()._perform_checks()
        if self.centralised or not self.input_has_agent_dim:
            raise ValueError("HetControlMlpHierarchical must be decentralized.")


# ==========================================================
# Config
# ==========================================================
@dataclass
class HetControlMlpHierarchicalConfig(ModelConfig):
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING
    desired_snd: float = MISSING
    tau: float = MISSING
    bootstrap_from_desired_snd: bool = MISSING
    process_shared: bool = MISSING
    probabilistic: Optional[bool] = MISSING
    scale_mapping: Optional[str] = MISSING

    n_subteams: int = 3
    subteam_tau: float = 0.1
    use_hard_assignment: bool = False
    selective_subteam_sharing: bool = False

    subteam_weight_init: float = 0.25
    agent_weight_init: float = 0.25

    normalize_weights: bool = False
    clip_weights: bool = True

    @staticmethod
    def associated_class():
        return HetControlMlpHierarchical
