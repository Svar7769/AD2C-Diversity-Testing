"""
Render callback for Multi-Agent Navigation with SND visualization.
"""
import matplotlib
matplotlib.use('Agg')  # Prevent display issues in headless environments

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
    policy = experiment.group_policies["agents"]
    model = get_het_model(policy)
    env_index = 0
    
    def snd(pos):
        """
        Compute SND (behavioral diversity) at given position(s).
        
        Args:
            pos: Position(s) to evaluate, shape [n_positions, 2]
            
        Returns:
            SND values, shape [n_positions, 1]
        """
        # Convert position to tensor on correct device
        pos_tensor = torch.as_tensor(
            pos, 
            device=model.device, 
            dtype=torch.float32
        )
        
        # Get observations at these positions
        obs = env.scenario.observation_from_pos(pos_tensor, env_index=env_index)
        obs = obs.view(-1, env.n_agents, obs.shape[-1])
        
        # Create observation tensor dict
        obs_td = TensorDict(
            {"agents": TensorDict({"observation": obs}, obs.shape[:2])}, 
            obs.shape[:1],
            device=model.device
        )
        
        # Compute agent actions (no gradient needed for visualization)
        with torch.no_grad():
            agent_actions = []
            for i in range(model.n_agents):
                action = model._forward(
                    obs_td, 
                    agent_index=i,
                    update_estimate=False  # Don't update running SND estimate
                ).get(model.out_key)
                agent_actions.append(action)
        
        # Compute pairwise behavioral distances
        pairwise_distances = compute_behavioral_distance(
            agent_actions,
            just_mean=True
        )
        
        # Average across all agent pairs
        avg_distance = pairwise_distances.mean(dim=-1)
        
        # Return as [n_positions, 1] on CPU for renderer
        return avg_distance.view(-1, 1).cpu()
    
    # Render with error handling
    try:
        return env.render(
            mode="rgb_array",
            visualize_when_rgb=False,
            plot_position_function=snd,
            plot_position_function_range=1.5,
            plot_position_function_cmap_alpha=0.5,
            plot_position_function_precision=0.05,  # 0.05 for quality, 0.1 for speed
            plot_position_function_cmap_range=[0.0, 1.0],
            env_index=env_index
        )
    except Exception as e:
        # Don't crash the experiment if rendering fails
        print(f"\n⚠️  [Render Warning] Visualization failed: {e}\n")
        return None