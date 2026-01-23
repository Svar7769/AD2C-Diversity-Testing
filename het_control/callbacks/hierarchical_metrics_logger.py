from typing import Optional, Dict, List
import torch
import torch.nn.functional as F
from benchmarl.experiment.callback import Callback
from tensordict import TensorDictBase

from het_control.callbacks.callback import get_het_model
from het_control.callbacks.snd import compute_behavioral_distance, compute_statistical_distance
from het_control.models.het_control_mlp_hierarchical import HetControlMlpHierarchical
from het_control.models.utils import squash

class HierarchicalMetricsLoggerCallback(Callback):
    """Unified logging for hierarchical models using reference SND functions.
    
    Updated to match SndCallback by computing distances in Action Space (Post-Tanh).
    """
    
    def __init__(self, control_group: str = "agents", log_interval: int = 100):
        super().__init__()
        self.control_group = control_group
        self.log_interval = log_interval
        self.iteration = 0
        self.model: Optional[HetControlMlpHierarchical] = None
        self.one_agent_per_subteam = False
        
    def on_setup(self):
        try:
            policy = self.experiment.group_policies.get(self.control_group)
            if policy is None: return
            model = get_het_model(policy)
            if isinstance(model, HetControlMlpHierarchical):
                self.model = model
                self.one_agent_per_subteam = (model.n_agents == model.n_subteams)
                print(f"✅ HierarchicalMetrics initialized using reference SND utilities")
            else:
                self.model = None
        except Exception as e:
            print(f"⚠️  HierarchicalMetrics init failed: {e}")
            self.model = None
            
    def on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        return batch

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        """Compute diversity metrics and team assignments on evaluation rollouts."""
        if self.model is None or not rollouts:
            return
        if not len(self.experiment.group_map[self.control_group]) > 1:
            return

        try:
            with torch.no_grad():
                obs = torch.cat(
                    [
                        rollout.select((self.control_group, "observation")).get(
                            (self.control_group, "observation")
                        )
                        for rollout in rollouts
                    ],
                    dim=0,
                )
                metrics = {}

                # Weights
                subteam_w, agent_w = self.model.get_hierarchical_weights()
                metrics.update({
                    "weights/subteam": subteam_w.item() if torch.is_tensor(subteam_w) else subteam_w,
                    "weights/agent": agent_w.item() if torch.is_tensor(agent_w) else agent_w,
                })

                # Team assignment logging
                assignments = self.model.compute_subteam_assignment(obs)  # [B, N, K]
                if assignments.dim() > 2:
                    assignments_mean = assignments.mean(dim=0)  # [N, K]
                else:
                    assignments_mean = assignments
                dominant = torch.argmax(assignments_mean, dim=-1)  # [N]

                flat_assignments = assignments.reshape(-1, assignments.shape[-1])
                probs = flat_assignments.clamp(min=1e-8)
                entropy = (-probs * probs.log()).sum(dim=-1).mean().item()
                metrics["assignments/entropy"] = entropy
                metrics["assignments/mean_max_prob"] = probs.max(dim=-1).values.mean().item()

                counts = torch.bincount(
                    torch.argmax(flat_assignments, dim=-1),
                    minlength=self.model.n_subteams,
                ).float()
                total = counts.sum().clamp(min=1.0)
                for s in range(self.model.n_subteams):
                    metrics[f"assignments/subteam_{s}_fraction"] = (counts[s] / total).item()
                for i in range(assignments_mean.shape[0]):
                    metrics[f"assignments/agent_{i}_subteam"] = int(dominant[i].item())
                    metrics[f"assignments/agent_{i}_max_prob"] = assignments_mean[i].max().item()
                    for s in range(assignments_mean.shape[1]):
                        metrics[f"assignments/agent_{i}_prob_{s}"] = assignments_mean[i, s].item()

                # Actions and diversity (match SndCallback path)
                from tensordict import TensorDict
                obs_td = TensorDict({self.model.in_key: obs}, batch_size=obs.shape[:-1])
                out_td = self.model._forward(obs_td, agent_index=None, compute_estimate=False)
                actions = out_td.get(self.model.out_key)
                agent_outputs_actions = list(actions.unbind(dim=-2))
                just_mean = True

                overall_snd_tensor = compute_behavioral_distance(agent_outputs_actions, just_mean)
                metrics["diversity/overall_snd"] = overall_snd_tensor.mean().item()

                # Component norms to diagnose growth
                obs_norm = self.model.input_norm(obs)
                shared_out = self.model.shared_mlp(obs_norm)
                if self.model.probabilistic:
                    mu_shared, _ = self.model.scale_extractor(shared_out)
                else:
                    mu_shared = shared_out
                if self.model.process_shared:
                    mu_shared = squash(
                        mu_shared,
                        self.model.action_spec[self.model.agent_group, "action"],
                        clamp=False,
                    )

                mu_agent = self.model.agent_mlps(obs_norm)
                logits = self.model.assignment_net(obs_norm)
                if self.model.use_hard_assignment:
                    assign_w = F.one_hot(logits.argmax(-1), self.model.n_subteams).float()
                else:
                    assign_w = torch.softmax(logits / self.model.subteam_tau, dim=-1)
                sub_outs = torch.stack([mlp(obs_norm) for mlp in self.model.subteam_mlps], dim=-2)
                mu_subteam = torch.einsum("...nka,...nk->...na", sub_outs, assign_w)
                scaling = self.model._get_snd_scaling(mu_agent, update=False, compute=False)

                weighted_subteam = subteam_w * mu_subteam
                weighted_agent = agent_w * mu_agent
                deviation = scaling * (weighted_subteam + weighted_agent)

                metrics["components/shared_norm"] = mu_shared.norm(dim=-1).mean().item()
                metrics["components/deviation_norm"] = deviation.norm(dim=-1).mean().item()
                metrics["components/subteam_norm"] = weighted_subteam.norm(dim=-1).mean().item()
                metrics["components/agent_norm"] = weighted_agent.norm(dim=-1).mean().item()

                if not self.one_agent_per_subteam:
                    intra_snd, inter_snd = self._compute_dlbc_snd_reference(
                        agent_outputs_actions, dominant, just_mean
                    )
                    metrics["diversity/intra_group_snd"] = intra_snd
                    metrics["diversity/inter_group_snd"] = inter_snd
                    if intra_snd > 1e-8:
                        metrics["diversity/inter_intra_ratio"] = inter_snd / intra_snd
                else:
                    metrics["diversity/inter_group_snd"] = metrics["diversity/overall_snd"]

                if metrics:
                    self.experiment.logger.log(metrics, step=self.experiment.n_iters_performed)
        except Exception as e:
            print(f"⚠️  HierarchicalMetrics eval error: {e}")
            import traceback; traceback.print_exc()

    def _compute_dlbc_snd_reference(self, agent_outputs, agent_subteams, just_mean) -> tuple:
        """Computes DLBC metrics using reference compute_statistical_distance."""
        n_subteams = self.model.n_subteams
        subteam_agents = {s: [] for s in range(n_subteams)}
        for idx, s in enumerate(agent_subteams.tolist()):
            subteam_agents[s].append(idx)
        
        # Intra-group calculation
        intra_vals = []
        for s in range(n_subteams):
            group = subteam_agents[s]
            if len(group) >= 2:
                group_out = [agent_outputs[i] for i in group]
                d = compute_behavioral_distance(group_out, just_mean)
                intra_vals.append(d.mean().item())
        
        # Inter-group calculation
        inter_vals = []
        for s1 in range(n_subteams):
            for s2 in range(s1 + 1, n_subteams):
                g1, g2 = subteam_agents[s1], subteam_agents[s2]
                for i in g1:
                    for j in g2:
                        d_pair = compute_statistical_distance(agent_outputs[i], agent_outputs[j], just_mean)
                        inter_vals.append(d_pair.mean().item())
        
        intra_snd = sum(intra_vals) / len(intra_vals) if intra_vals else 0.0
        inter_snd = sum(inter_vals) / len(inter_vals) if inter_vals else 0.0
        return intra_snd, inter_snd

    def _get_observation(self, batch, group):
        for key in [(group, "observation"), (group, "obs"), "observation"]:
            if key in batch.keys(True): return batch.get(key)
        return None

    # Note: action computation uses model._forward to match SndCallback exactly.
