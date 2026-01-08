from typing import List
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import networkx as nx

from het_control.callback import get_het_model
from het_control.snd import compute_behavioral_distance
from benchmarl.experiment.callback import Callback
from tensordict._td import TensorDict  # Required for the Graph Visualizer

class SNDHeatmapVisualizer:
    def __init__(self, key_name="Visuals/SND_Heatmap"):
        self.key_name = key_name

    def generate(self, snd_matrix, step_count):
        # snd_matrix is now GUARANTEED to be a clean 2D Numpy array
        n_agents = snd_matrix.shape[0]
        agent_labels = [f"Agent {i+1}" for i in range(n_agents)]
        
        # Calculate SND value
        iu = np.triu_indices(n_agents, k=1)
        if len(iu[0]) > 0:
            snd_value = float(np.mean(snd_matrix[iu]))
        else:
            snd_value = 0.0

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
                # Dynamic text color for visibility
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
        
        # Create pairs i < j
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
    """
    Manages the individual visualizers and handles ALL data cleaning centrally.
    """
    def __init__(self):
        self.visualizers = [
            SNDHeatmapVisualizer(),
            SNDBarChartVisualizer(),
            SNDGraphVisualizer()
        ]

    def _prepare_matrix(self, snd_matrix):
        """
        Robustly converts and reshapes matrix.
        Handles non-square pairwise distance outputs.
        """
        # 1. Convert to Numpy
        if hasattr(snd_matrix, "detach"):
            snd_matrix = snd_matrix.detach().cpu().numpy()
        elif not isinstance(snd_matrix, np.ndarray):
            snd_matrix = np.array(snd_matrix)

        # 2. "Peel" dimensions until we hit 2D or 1D
        while snd_matrix.ndim > 2:
            snd_matrix = snd_matrix[0]

        # 3. Handle non-square matrix from pairwise distances
        if snd_matrix.ndim == 2 and snd_matrix.shape[0] != snd_matrix.shape[1]:
            # This is likely a [n_agents, n_pairs] or [n_pairs, n_agents] format
            # We need to reconstruct the full symmetric matrix
            
            # Determine n_agents from the shape
            n_agents = min(snd_matrix.shape)
            n_pairs_expected = n_agents * (n_agents - 1) // 2
            
            # Check if one dimension matches expected pairs
            if n_pairs_expected in snd_matrix.shape:
                # Flatten to get pairwise distances
                if snd_matrix.shape[1] == n_pairs_expected:
                    pairwise_flat = snd_matrix[0, :]  # Take first row
                else:
                    pairwise_flat = snd_matrix[:, 0]  # Take first column
                
                # Reconstruct symmetric matrix from pairwise distances
                full_matrix = np.zeros((n_agents, n_agents))
                idx = 0
                for i in range(n_agents):
                    for j in range(i + 1, n_agents):
                        full_matrix[i, j] = pairwise_flat[idx]
                        full_matrix[j, i] = pairwise_flat[idx]
                        idx += 1
                snd_matrix = full_matrix
            else:
                # Fallback: just take the square portion
                n = min(snd_matrix.shape)
                snd_matrix = snd_matrix[:n, :n]

        # 4. Handle 1D edge case
        if snd_matrix.ndim == 1:
            size = snd_matrix.shape[0]
            n_agents = int(np.sqrt(size))
            if n_agents * n_agents == size:
                snd_matrix = snd_matrix.reshape(n_agents, n_agents)
            else:
                # Reconstruct from pairwise distances
                # Calculate n_agents from n_pairs: n_pairs = n*(n-1)/2
                # Solving: n^2 - n - 2*n_pairs = 0
                n_agents = int((1 + np.sqrt(1 + 8 * size)) / 2)
                full_matrix = np.zeros((n_agents, n_agents))
                idx = 0
                for i in range(n_agents):
                    for j in range(i + 1, n_agents):
                        if idx < size:
                            full_matrix[i, j] = snd_matrix[idx]
                            full_matrix[j, i] = snd_matrix[idx]
                            idx += 1
                snd_matrix = full_matrix

        # 5. Create copy
        snd_matrix = snd_matrix.copy()

        # 6. Ensure symmetry (only if already square)
        if snd_matrix.shape[0] == snd_matrix.shape[1]:
            snd_matrix = (snd_matrix + snd_matrix.T) / 2.0

        # 7. Set diagonals to zero
        n = snd_matrix.shape[0]
        if n > 0:
            for i in range(n):
                snd_matrix[i, i] = 0.0
        
        return snd_matrix

    def generate_all(self, snd_matrix, step_count):
        # Clean the matrix ONCE here
        clean_matrix = self._prepare_matrix(snd_matrix)
        
        all_plots = {}
        for visualizer in self.visualizers:
            try:
                # Pass the clean matrix to all visualizers
                plots = visualizer.generate(clean_matrix, step_count)
                all_plots.update(plots)
            except Exception as e:
                print(f"Error generating {visualizer.__class__.__name__}: {e}")
                # Optional: Print shape to help debug if it fails again
                print(f"Failed Matrix Shape: {clean_matrix.shape}")
        return all_plots
    
class SNDVisualizerCallback(Callback):
    """
    Computes the SND matrix and uses the Manager to log visualizations.
    """
    def __init__(self):
        super().__init__()
        self.control_group = None
        self.model = None
        # Initialize the manager that holds the 3 plot classes
        self.viz_manager = SNDVisualizationManager()

    def on_setup(self):
        """Auto-detects the agent group and initializes the model wrapper."""
        if not self.experiment.group_policies:
            print("\nWARNING: No group policies found. SND Visualizer disabled.\n")
            return

        self.control_group = list(self.experiment.group_policies.keys())[0]
        policy = self.experiment.group_policies[self.control_group]
        
        # Ensure 'get_het_model' is imported or available in this scope
        self.model = get_het_model(policy)

        if self.model is None:
             print(f"\nWARNING: Could not extract HetModel for group '{self.control_group}'. Visualizer disabled.\n")

    def _get_agent_actions_for_rollout(self, rollout):
        """Helper to run the forward pass and get actions for SND computation."""
        obs = rollout.get((self.control_group, "observation"))
        actions = []
        for i in range(self.model.n_agents):
            temp_td = TensorDict(
                {(self.control_group, "observation"): obs},
                batch_size=obs.shape[:-1]
            )
            action_td = self.model._forward(temp_td, agent_index=i, compute_estimate=False)
            actions.append(action_td.get(self.model.out_key))
        return actions

    def on_evaluation_end(self, rollouts: List[TensorDict]):
        """Runs at the end of evaluation to compute SND and log plots."""
        if self.model is None:
            return

        logs_to_push = {}
        first_rollout_snd_matrix = None

        with torch.no_grad():
            for i, r in enumerate(rollouts):
                # We only need the matrix from the first rollout for clean visualization
                if i > 0: 
                    break

                agent_actions = self._get_agent_actions_for_rollout(r)
                
                # Ensure 'compute_behavioral_distance' is imported/available
                pairwise_distances_tensor = compute_behavioral_distance(agent_actions, just_mean=False)
                
                if pairwise_distances_tensor.ndim > 2:
                    pairwise_distances_tensor = pairwise_distances_tensor.mean(dim=0)

                first_rollout_snd_matrix = pairwise_distances_tensor.cpu().numpy()

        # Generate and Log Visualizations via the Manager
        if first_rollout_snd_matrix is not None:
            visual_logs = self.viz_manager.generate_all(
                snd_matrix=first_rollout_snd_matrix, 
                step_count=self.experiment.n_iters_performed
            )
            logs_to_push.update(visual_logs)
            
            # Update the logger
            self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)