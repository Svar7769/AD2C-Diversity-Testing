from typing import List
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import networkx as nx

from het_control.callbacks.callback import get_het_model
from het_control.callbacks.snd import compute_behavioral_distance
from benchmarl.experiment.callback import Callback
from tensordict import TensorDictBase


class SNDHeatmapVisualizer:
    def __init__(self, key_name="Visuals/SND_Heatmap"):
        self.key_name = key_name

    def generate(self, snd_matrix, step_count):
        n_agents = snd_matrix.shape[0]
        agent_labels = [f"Agent {i+1}" for i in range(n_agents)]
        
        # Calculate SND value from upper triangle (excluding diagonal)
        iu = np.triu_indices(n_agents, k=1)
        snd_value = float(np.mean(snd_matrix[iu])) if len(iu[0]) > 0 else 0.0

        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(
            snd_matrix,
            cmap="viridis",
            interpolation="nearest",
            vmin=0, vmax=3 
        )

        ax.set_title(f"SND: {snd_value:.3f}  –  Step {step_count}")

        ax.set_xticks(np.arange(n_agents))
        ax.set_yticks(np.arange(n_agents))
        ax.set_xticklabels(agent_labels)
        ax.set_yticklabels(agent_labels)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        fig.colorbar(im, ax=ax, label="Distance")

        for i in range(n_agents):
            for j in range(n_agents):
                val = snd_matrix[i, j]
                text_color = "white" if val < 1.0 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color=text_color,
                    fontsize=9, fontweight="bold"
                )

        plt.tight_layout()
        img = wandb.Image(fig)
        plt.close(fig)
        return {self.key_name: img}


class SNDBarChartVisualizer:
    def __init__(self, key_name="Visuals/SND_BarChart"):
        self.key_name = key_name

    def generate(self, snd_matrix, step_count):
        n_agents = snd_matrix.shape[0]
        
        pairs = [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]
        if not pairs:
            return {}

        pair_values = [float(snd_matrix[i, j]) for i, j in pairs]
        pair_labels = [f"A{i+1}-A{j+1}" for i, j in pairs]

        snd_value = float(np.mean(pair_values))

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(pair_labels, pair_values, color="teal")

        ax.set_title(f"SND: {snd_value:.3f}  –  Step {step_count}")
        ax.set_ylabel("Distance")
        ax.set_ylim(0, 3)
        ax.tick_params(axis="x", rotation=45)

        ax.bar_label(bars, fmt="%.2f", padding=3)

        plt.tight_layout()
        img = wandb.Image(fig)
        plt.close(fig)
        return {self.key_name: img}


class SNDGraphVisualizer:
    def __init__(self, key_name="Visuals/SND_NetworkGraph"):
        self.key_name = key_name

    def generate(self, snd_matrix, step_count):
        n_agents = snd_matrix.shape[0]

        pairs = [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]
        if not pairs:
            return {}

        pair_values = [float(snd_matrix[i, j]) for i, j in pairs]
        snd_value = float(np.mean(pair_values))

        fig = plt.figure(figsize=(7, 7))
        G = nx.Graph()

        for i, j in pairs:
            G.add_edge(i, j, weight=float(snd_matrix[i, j]))

        pos = nx.spring_layout(G, seed=42)
        weights = [G[u][v]['weight'] for u, v in G.edges()]

        nx.draw_networkx_nodes(G, pos, node_size=750, node_color='lightblue')
        
        label_mapping = {i: f"A{i+1}" for i in range(n_agents)}
        nx.draw_networkx_labels(G, pos, labels=label_mapping, font_size=12, font_weight='bold')

        edges = nx.draw_networkx_edges(
            G, pos,
            edge_color=weights,
            edge_cmap=plt.cm.viridis,
            width=2,
            edge_vmin=0, edge_vmax=3
        )

        edge_labels = {(i, j): f"{snd_matrix[i, j]:.2f}" for i, j in pairs}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels,
            font_color='black', font_size=9, font_weight='bold'
        )

        plt.colorbar(edges, label='Distance')
        plt.title(f"SND: {snd_value:.3f}  –  Step {step_count}", fontsize=14)
        plt.axis('off')

        img = wandb.Image(fig)
        plt.close(fig)
        return {self.key_name: img}


class SNDVisualizationManager:
    """Manages individual visualizers and handles data cleaning centrally."""
    
    def __init__(self):
        self.visualizers = [
            SNDHeatmapVisualizer(),
            SNDBarChartVisualizer(),
            SNDGraphVisualizer()
        ]

    def _prepare_matrix(self, agent_actions):
        """
        Computes SND matrix from agent actions using the EXACT same method as SNDCallback.
        
        Args:
            agent_actions: List of action tensors, one per agent
                          Each tensor has shape [*batch, action_features]
            
        Returns:
            Symmetric 2D numpy array representing the full distance matrix
        """
        n_agents = len(agent_actions)
        
        # Use compute_behavioral_distance with just_mean=False (same as model's compute_estimate)
        # This returns shape [*batch, n_pairs] where n_pairs = n_agents*(n_agents-1)/2
        pairwise_distances = compute_behavioral_distance(agent_actions, just_mean=False)
        
        # Average over batch dimension to get single distance per pair
        # Result shape: [n_pairs]
        pairwise_distances_avg = pairwise_distances.mean(dim=tuple(range(pairwise_distances.ndim - 1)))
        
        # Convert to numpy
        pairwise_distances_np = pairwise_distances_avg.detach().cpu().numpy()
        
        # Reconstruct the full symmetric matrix from the flattened upper triangular pairs
        # The compute_behavioral_distance function iterates as: for i in range(n_agents), for j in range(i+1, n_agents)
        distance_matrix = np.zeros((n_agents, n_agents))
        
        pair_idx = 0
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                distance = pairwise_distances_np[pair_idx]
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # Make symmetric
                pair_idx += 1
        
        # Diagonal should be zero (distance from agent to itself)
        np.fill_diagonal(distance_matrix, 0.0)
        
        return distance_matrix

    def generate_all(self, agent_actions, step_count):
        """
        Generate all visualizations from agent actions.
        
        Args:
            agent_actions: List of action tensors from each agent
            step_count: Current training step
            
        Returns:
            Dictionary of plot names to wandb.Image objects
        """
        # Compute the distance matrix using the exact same method as SNDCallback
        clean_matrix = self._prepare_matrix(agent_actions)
        
        all_plots = {}
        for visualizer in self.visualizers:
            try:
                plots = visualizer.generate(clean_matrix, step_count)
                all_plots.update(plots)
            except Exception as e:
                print(f"Error generating {visualizer.__class__.__name__}: {e}")
                print(f"Matrix shape: {clean_matrix.shape}")
        
        return all_plots


class SNDVisualizerCallback(Callback):
    """
    Computes SND matrix using the EXACT same method as SNDCallback and logs visualizations.
    """
    
    def __init__(self):
        super().__init__()
        self.control_group = None
        self.model = None
        self.viz_manager = SNDVisualizationManager()

    def on_setup(self):
        """Auto-detect agent group and initialize model."""
        if not self.experiment.group_policies:
            print("\nWARNING: No group policies found. SND Visualizer disabled.\n")
            return

        # Use the first group (or you can specify which group to visualize)
        self.control_group = list(self.experiment.group_policies.keys())[0]
        policy = self.experiment.group_policies[self.control_group]
        
        self.model = get_het_model(policy)

        if self.model is None:
            print(f"\nWARNING: Could not extract HetModel for group '{self.control_group}'. Visualizer disabled.\n")
        else:
            print(f"\nSUCCESS: SND Visualizer initialized for group '{self.control_group}'.\n")

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        """
        Compute SND and generate visualizations at evaluation end.
        Uses the EXACT same computation method as SNDCallback.
        """
        if self.model is None or not rollouts:
            return

        # Only compute for groups with multiple agents (same check as SNDCallback)
        if not len(self.experiment.group_map[self.control_group]) > 1:
            return

        with torch.no_grad():
            # Concatenate observations over time from all rollouts
            # This matches the SNDCallback approach EXACTLY
            obs = torch.cat(
                [rollout.select((self.control_group, "observation")) for rollout in rollouts],
                dim=0
            )  # Shape: [*batch_size, n_agents, n_features]
            
            # Compute actions that each agent would take for these observations
            # EXACT same method as SNDCallback
            agent_actions = []
            for i in range(self.model.n_agents):
                action = self.model._forward(
                    obs, 
                    agent_index=i, 
                    compute_estimate=False
                ).get(self.model.out_key)
                agent_actions.append(action)
            
            # Generate visualizations using the manager
            # The manager will compute distances using compute_behavioral_distance
            # with just_mean=False, then reconstruct the full symmetric matrix
            visual_logs = self.viz_manager.generate_all(
                agent_actions=agent_actions,
                step_count=self.experiment.n_iters_performed
            )
            
            if visual_logs:
                self.experiment.logger.log(visual_logs, step=self.experiment.n_iters_performed)