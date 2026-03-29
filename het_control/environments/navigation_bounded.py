"""
navigation_bounded.py
─────────────────────
Custom VMAS navigation scenario for Anki Vector sim-to-real training.

Features vs. vanilla navigation:
  - Bounded arena  (x_semidim / y_semidim walls)
  - Physical agent-agent collisions  (agents bounce off each other + penalty)
  - No lidar sensors  → obs_dim stays small
  - Each agent observes relative positions of ALL other agents
  - obs_dim = 4 + 2 + 2*(n_agents-1)
            = [x, y, vx, vy, dx_goal, dy_goal, dx_a1, dy_a1, dx_a2, dy_a2, ...]
            = 10  for 3 agents

Usage in BenchMARL config:
  task: vmas/navigation_bounded
"""

import typing
from typing import Callable, Dict, List

import torch
from torch import Tensor

from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents        = kwargs.pop("n_agents", 3)
        self.agents_with_same_goal = kwargs.pop("agents_with_same_goal", 1)
        self.observe_all_goals = kwargs.pop("observe_all_goals", False)
        self.shared_rew      = kwargs.pop("shared_rew", False)

        self.world_spawning_x = kwargs.pop("world_spawning_x", 1.0)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 1.0)

        self.agent_radius    = kwargs.pop("agent_radius", 0.1)
        self.max_steps       = kwargs.pop("max_steps", 500)

        self.pos_shaping_factor    = kwargs.pop("pos_shaping_factor", 1.0)
        self.final_reward          = kwargs.pop("final_reward", 0.01)
        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1.0)
        self.wall_collision_penalty  = kwargs.pop("wall_collision_penalty", 0.0)
        # Agents are considered "at the wall" when their centre is within this
        # distance of the boundary.  Defaults to agent_radius (touching wall).
        self.wall_collision_margin   = kwargs.pop("wall_collision_margin", None)

        # wall_collision_margin defaults to agent_radius (resolved after agent_radius is set)
        if self.wall_collision_margin is None:
            self.wall_collision_margin = self.agent_radius

        # Consume any leftover kwargs so VMAS doesn't complain
        kwargs.pop("collisions", None)
        kwargs.pop("enforce_bounds", None)
        kwargs.pop("agent_collision", None)
        kwargs.pop("split_goals", None)
        kwargs.pop("lidar_range", None)
        kwargs.pop("n_lidar_rays", None)
        kwargs.pop("comms_range", None)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = 0.005

        # Bounded arena — matches the 2 m × 2 m Gazebo arena (±1 m)
        world = World(
            batch_dim,
            device,
            substeps=2,
            x_semidim=self.world_spawning_x,
            y_semidim=self.world_spawning_y,
        )

        known_colors = [
            (0.22, 0.49, 0.72),
            (1.00, 0.50, 0.00),
            (0.30, 0.69, 0.29),
            (0.97, 0.51, 0.75),
            (0.60, 0.31, 0.64),
            (0.89, 0.10, 0.11),
        ]

        # Add agents — collide=True → physical bouncing, no sensors
        for i in range(self.n_agents):
            color = known_colors[i] if i < len(known_colors) else (0.5, 0.5, 0.5)
            agent = Agent(
                name=f"agent_{i}",
                collide=True,           # physical collisions enabled
                color=color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=None,           # no lidar
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = torch.zeros(batch_dim, device=device)
            agent.wall_collision_rew = torch.zeros(batch_dim, device=device)
            world.add_agent(agent)

            goal = Landmark(
                name=f"goal_{i}",
                collide=False,
                color=color,
            )
            world.add_landmark(goal)
            agent.goal = goal

        self.pos_rew  = torch.zeros(batch_dim, device=device)
        self.final_rew = torch.zeros(batch_dim, device=device)

        return world

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (-self.world_spawning_x, self.world_spawning_x),
            (-self.world_spawning_y, self.world_spawning_y),
        )

        occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)

        goal_poses = []
        for _ in self.world.agents:
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=self.world,
                min_dist_between_entities=self.min_distance_between_entities,
                x_bounds=(-self.world_spawning_x, self.world_spawning_x),
                y_bounds=(-self.world_spawning_y, self.world_spawning_y),
            )
            goal_poses.append(position.squeeze(1))
            occupied_positions = torch.cat([occupied_positions, position], dim=1)

        for i, agent in enumerate(self.world.agents):
            goal_index = 0 if i < self.agents_with_same_goal else i
            agent.goal.set_pos(goal_poses[goal_index], batch_index=env_index)

            if env_index is None:
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos, dim=1
                    ) * self.pos_shaping_factor
                )
            else:
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    ) * self.pos_shaping_factor
                )

    # ── Reward ─────────────────────────────────────────────────────────────────

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for a in self.world.agents:
                self.pos_rew += self._agent_reward(a)
                a.agent_collision_rew[:] = 0
                a.wall_collision_rew[:] = 0

            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1), dim=-1
            )
            self.final_rew[self.all_goal_reached] = self.final_reward

            # Agent-agent collision penalty
            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        dist = self.world.get_distance(a, b)
                        penalty = (dist <= self.min_collision_distance).float() * self.agent_collision_penalty
                        a.agent_collision_rew += penalty
                        b.agent_collision_rew += penalty

            # Wall collision penalty — fires when agent centre is within
            # wall_collision_margin of any boundary (x or y axis)
            if self.wall_collision_penalty != 0.0:
                margin = self.wall_collision_margin
                for a in self.world.agents:
                    at_wall = (
                        (a.state.pos[:, 0].abs() >= self.world_spawning_x - margin)
                        | (a.state.pos[:, 1].abs() >= self.world_spawning_y - margin)
                    ).float()
                    a.wall_collision_rew += at_wall * self.wall_collision_penalty

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew + agent.wall_collision_rew

    def _agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos, dim=-1
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        return agent.pos_rew

    # ── Observation ────────────────────────────────────────────────────────────

    def observation(self, agent: Agent) -> Tensor:
        """
        obs = [x, y, vx, vy, dx_to_own_goal, dy_to_own_goal,
               dx_to_other_agent_0, dy_to_other_agent_0, ...]

        obs_dim = 4 + 2 + 2*(n_agents-1)
                = 10  for 3 agents
        """
        obs = [agent.state.pos, agent.state.vel]

        # Goal delta (own goal only — matches checkpoint format)
        obs.append(agent.state.pos - agent.goal.state.pos)

        # Relative positions of all other agents (for collision avoidance)
        for other in self.world.agents:
            if other is agent:
                continue
            obs.append(agent.state.pos - other.state.pos)

        return torch.cat(obs, dim=-1)

    # ── Done ───────────────────────────────────────────────────────────────────

    def done(self):
        return torch.stack(
            [
                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos, dim=-1
                ) < agent.shape.radius
                for agent in self.world.agents
            ],
            dim=-1,
        ).all(-1)

    # ── Info ───────────────────────────────────────────────────────────────────

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collisions": agent.agent_collision_rew,
            "wall_collisions": agent.wall_collision_rew,
        }