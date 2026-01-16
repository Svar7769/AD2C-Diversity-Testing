import matplotlib
matplotlib.use('Agg') # MUST BE FIRST

import torch
import warnings
from tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase

import matplotlib
matplotlib.use('Agg')

from het_control.callbacks.callback import get_het_model
from het_control.callbacks.snd import compute_behavioral_distance

def render_callback(experiment, env: EnvBase, data: TensorDictBase):
    policy = experiment.group_policies["agents"]
    model = get_het_model(policy)
    env_index = 0

    def snd(pos):
        # Move pos to device immediately
        pos_tensor = torch.as_tensor(pos, device=model.device, dtype=torch.float)
        
        # Get observations for the given positions
        obs = env.scenario.observation_from_pos(pos_tensor, env_index=env_index)
        obs = obs.view(-1, env.n_agents, obs.shape[-1])
        
        obs_td = TensorDict(
            {"agents": TensorDict({"observation": obs}, obs.shape[:2])}, 
            obs.shape[:1],
            device=model.device
        )

        # OPTIMIZATION: Call forward ONCE for all agents instead of looping
        # In HetControlMlpEmpirical, leaving agent_index=None computes all agents
        with torch.no_grad():
            # We don't want to update the running estimate of SND during rendering
            out_td = model._forward(obs_td, agent_index=None, update_estimate=False)
            
            # The model forward logic for HetControlMlpEmpirical usually 
            # computes the agents' individual deviations within the forward pass.
            # If your model needs the list of individual agent actions for compute_behavioral_distance:
            agent_actions = []
            # We use the agent_networks directly to get the deviations 'agent_out'
            for agent_net in model.agent_mlps.agent_networks:
                agent_actions.append(agent_net(obs))
        
        pairwise_distances = compute_behavioral_distance(
            agent_actions,
            just_mean=True,
        )
        
        avg_distance = pairwise_distances.mean(dim=-1)
        return avg_distance.view(-1, 1).cpu() # Move back to CPU for the renderer

    # Wrap in try-except to prevent the "Aborted (core dumped)" from killing the whole experiment
    try:
        return env.render(
            mode="rgb_array",
            visualize_when_rgb=False,
            plot_position_function=snd,
            plot_position_function_range=1.5,
            plot_position_function_cmap_alpha=0.5,
            env_index=env_index,
            plot_position_function_precision=0.1, # Increased precision to 0.1 for speed
            plot_position_function_cmap_range=[0.0, 1.0],
        )
    except Exception as e:
        print(f"\n[Render Warning] Visualization failed: {e}\n")
        return None