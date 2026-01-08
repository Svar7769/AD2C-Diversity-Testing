# Copyright (c) 2024.
# ProrokLab (https://www.proroklab.org/)
# All rights reserved.

import torch
import wandb
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

from benchmarl.algorithms import MappoConfig, IppoConfig, MasacConfig, IsacConfig
from benchmarl.experiment import Experiment
from benchmarl.hydra_config import (
    load_algorithm_config_from_hydra,
    load_experiment_config_from_hydra,
    load_task_config_from_hydra,
    load_model_config_from_hydra,
)
from het_control.run import setup


def get_evaluation_experiment(cfg: DictConfig, task_name: str, algorithm_name: str) -> Experiment:
    """
    Sets up a BenchMARL experiment for evaluation only.
    """
    setup(task_name)
    print(f"\nSetting up evaluation for Algorithm: {algorithm_name}, Task: {task_name}")

    algorithm_config = load_algorithm_config_from_hydra(cfg.algorithm)
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)
    task_config = load_task_config_from_hydra(cfg.task, task_name)
    critic_model_config = load_model_config_from_hydra(cfg.critic_model)
    model_config = load_model_config_from_hydra(cfg.model)

    if isinstance(algorithm_config, (MappoConfig, IppoConfig, MasacConfig, IsacConfig)):
        model_config.probabilistic = True
        model_config.scale_mapping = algorithm_config.scale_mapping
        algorithm_config.scale_mapping = "relu"
    else:
        model_config.probabilistic = False

    experiment = Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=cfg.seed,
        config=experiment_config,
        callbacks=[],
    )
    return experiment


def hydra_experiment(cfg: DictConfig) -> None:
    """The main evaluation function with a manual evaluation loop and W&B logging."""
    task_name = "vmas/navigation"
    algorithm_name = "ippo"
    
    # 1. Initialize Weights & Biases
    wandb.init(
        project="AD2C_evaluation",
        name=f"eval_{task_name}_{algorithm_name}",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

    experiment = get_evaluation_experiment(cfg=cfg, task_name=task_name, algorithm_name=algorithm_name)
    
    print(f"\nLoading model from: {cfg.experiment.restore_file}")
    experiment.algorithm.load_state_dict_from_file(experiment.config.restore_file)
    
    task = experiment.task
    policy = experiment.algorithm.get_policy_for_collection("evaluation")

    print("Starting evaluation run...")
    tensordict = task.reset()
    
    for i in range(experiment.config.max_n_frames):
        task.render()
        with torch.no_grad():
            tensordict = policy(tensordict)
        tensordict = task.step(tensordict)

        # 2. Log metrics to wandb
        step_reward = tensordict[("next", "reward")].sum().item()
        wandb.log({"evaluation/step_reward": step_reward, "step": i})

    print("Evaluation finished.")
    # 3. Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    overrides = [
        "task=vmas/navigation",
        "algorithm=ippo",
        "model=hetcontrolmlpempirical",
        'experiment.restore_file="outputs/2025-10-18/18-32-09/AD2C_ippo_navigation__25_10_18-18_32_12/checkpoints/checkpoint_3000000.pt"',
        "experiment.max_n_frames=5000"
    ]

    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config", overrides=overrides)
        hydra_experiment(cfg)