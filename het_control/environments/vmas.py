"""
Render callback for Multi-Agent Navigation with SND visualization.
"""
import matplotlib
matplotlib.use('Agg')
import torch
from tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase
from het_control.callbacks.callback import get_het_model
from het_control.callbacks.snd import compute_behavioral_distance


def render_callback(experiment, env: EnvBase, data: TensorDictBase):
    """
    Render callback to visualize diversity distribution in Multi-Agent Navigation.
    
    Args:
        experiment: BenchMARL experiment instance
        env: Environment instance
        data: Tensor dictionary with episode data
    
    Returns:
        Rendered frame as RGB array, or None if rendering fails
    """
    # Get model once (outside nested function)
    policy = experiment.group_policies["agents"]
    model = get_het_model(policy)
    
    # Check if model supports SND visualization
    if model is None or not hasattr(model, '_forward'):
        print("\n⚠️  [Render Warning] Model doesn't support SND visualization\n")
        try:
            return env.render(mode="rgb_array", visualize_when_rgb=False)
        except Exception as e:
            print(f"\n⚠️  [Render Warning] Basic rendering failed: {e}\n")
            return None
    
    env_index = 0
    n_agents = env.n_agents
    
    def snd(pos):
        """
        Compute SND (behavioral diversity) at given position(s).
        
        Args:
            pos: Position(s) to evaluate, shape [n_positions, 2]
        
        Returns:
            SND values, shape [n_positions, 1]
        """
        try:
            # Convert position to tensor on correct device
            pos_tensor = torch.as_tensor(
                pos, 
                device=model.device, 
                dtype=torch.float32
            )
            
            # Handle single position vs. batch of positions
            if pos_tensor.dim() == 1:
                pos_tensor = pos_tensor.unsqueeze(0)  # [2] → [1, 2]
            
            n_positions = pos_tensor.shape[0]
            
            # Get observations at these positions
            # This returns [n_positions, obs_dim] where obs contains all agents' observations concatenated
            obs = env.scenario.observation_from_pos(pos_tensor, env_index=env_index)
            
            # ⭐ KEY FIX: Reshape to separate per-agent observations
            # obs shape is [n_positions, n_agents * obs_per_agent]
            # We need [n_positions, n_agents, obs_per_agent]
            obs_per_agent = obs.shape[-1] // n_agents
            obs = obs.view(n_positions, n_agents, obs_per_agent)
            
            # Create observation tensor dict with correct batch structure
            obs_td = TensorDict(
                {"agents": TensorDict(
                    {"observation": obs}, 
                    batch_size=[n_positions, n_agents]
                )}, 
                batch_size=[n_positions],
                device=model.device
            )
            
            # Compute agent actions for all positions (vectorized)
            with torch.no_grad():
                agent_actions = []
                for i in range(n_agents):
                    action = model._forward(
                        obs_td, 
                        agent_index=i,
                        update_estimate=False
                    ).get(model.out_key)
                    agent_actions.append(action)
                
                # Stack actions: [n_agents, n_positions, action_dim]
                agent_actions = torch.stack(agent_actions, dim=0)
                
                # Compute pairwise behavioral distances
                # Output: [n_positions] or [n_positions, n_pairs]
                pairwise_distances = compute_behavioral_distance(
                    agent_actions,
                    just_mean=True
                )
                
                # Ensure output is [n_positions]
                if pairwise_distances.dim() > 1:
                    avg_distance = pairwise_distances.mean(dim=-1)
                else:
                    avg_distance = pairwise_distances
                
                # Return as [n_positions, 1] numpy array
                return avg_distance.view(-1, 1).cpu().numpy()
                
        except Exception as e:
            # Silently return zeros on error (don't spam console)
            if isinstance(pos, torch.Tensor):
                n_pos = pos.shape[0] if pos.dim() > 1 else 1
            else:
                import numpy as np
                n_pos = len(pos) if hasattr(pos, '__len__') and len(np.array(pos).shape) > 1 else 1
            return torch.zeros(n_pos, 1).cpu().numpy()
    
    # Render with SND heatmap
    try:
        return env.render(
            mode="rgb_array",
            visualize_when_rgb=False,
            plot_position_function=snd,
            plot_position_function_range=1.5,
            plot_position_function_cmap_alpha=0.5,
            plot_position_function_precision=0.1,  # Use 0.1 for speed (30×30 grid)
            plot_position_function_cmap_range=[0.0, 1.0],
            env_index=env_index
        )
    except Exception as e:
        # Try basic rendering without SND overlay
        print(f"\n⚠️  [Render Warning] SND visualization failed: {e}\n")
        try:
            return env.render(mode="rgb_array", visualize_when_rgb=False, env_index=env_index)
        except Exception as e2:
            print(f"\n⚠️  [Render Warning] Even basic rendering failed: {e2}\n")
            return None