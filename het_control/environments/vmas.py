import torch
from tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase

from het_control.callbacks.callback import get_het_model
from het_control.callbacks.snd import compute_behavioral_distance


def render_callback(experiment, env: EnvBase, data: TensorDictBase):
    """
    Render callback used in the Multi-Agent Navigation scenario to visualize the
    diversity distribution under the evaluation rendering.

    """
    policy = experiment.group_policies["agents"]
    model = get_het_model(policy)
    env_index = 0

    def snd(pos):
        obs = env.scenario.observation_from_pos(
            torch.tensor(pos, device=model.device), env_index=env_index
        )
        n_pos = pos.shape[0]
        obs = obs.unsqueeze(1).expand(n_pos, env.n_agents, obs.shape[-1]).to(torch.float)
        obs_td = TensorDict(
            {"agents": TensorDict({"observation": obs}, obs.shape[:2])}, obs.shape[:1]
        )
        agent_actions = []
        for i in range(model.n_agents):
            action = model._forward(obs_td, agent_index=i).get(model.out_key)
            agent_actions.append(action[:, i:i+1, :])  # select only agent i's action
        distance = compute_behavioral_distance(
            agent_actions,
            just_mean=True,
        )
        # Average pairwise distances down to one scalar per position
        if distance.dim() > 1:
            distance = distance.mean(dim=tuple(range(1, distance.dim())))
        distance = distance.view(n_pos, 1)
        return distance

    return env.render(
        mode="rgb_array",
        visualize_when_rgb=False,
        plot_position_function=snd,
        plot_position_function_range=1.5,
        plot_position_function_cmap_alpha=0.5,
        env_index=env_index,
        plot_position_function_precision=0.05,
        plot_position_function_cmap_range=[0.0, 1.0],
    )