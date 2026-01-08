#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import sys
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from hydra.core.global_hydra import GlobalHydra

from het_control.callbacks.sndVisualCallback import SNDVisualizerCallback
import benchmarl.models
from benchmarl.algorithms import *
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.hydra_config import (
    load_algorithm_config_from_hydra,
    load_experiment_config_from_hydra,
    load_task_config_from_hydra,
    load_model_config_from_hydra,
)
from het_control.callback import *
from het_control.environments.vmas import render_callback
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpiricalConfig
from het_control.callbacks.esc_callback import ExtremumSeekingController
from het_control.callbacks.sndESLogger import TrajectorySNDLoggerCallback
# from het_control.callbacks.performaceLoggerCallback import performaceLoggerCallback



def setup(task_name):
    benchmarl.models.model_config_registry.update(
        {
            "hetcontrolmlpempirical": HetControlMlpEmpiricalConfig,
        }
    )
    if task_name == "vmas/balance":   #vmas/navigation
        # Set the render callback for the navigatio case study
        VmasTask.render_callback = render_callback


def get_experiment(cfg: DictConfig) -> Experiment:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    setup(task_name)

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    algorithm_config = load_algorithm_config_from_hydra(cfg.algorithm)
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)
    task_config = load_task_config_from_hydra(cfg.task, task_name)
    critic_model_config = load_model_config_from_hydra(cfg.critic_model)
    model_config = load_model_config_from_hydra(cfg.model)

    if isinstance(algorithm_config, (MappoConfig, IppoConfig, MasacConfig, IsacConfig)):
        model_config.probabilistic = True
        model_config.scale_mapping = algorithm_config.scale_mapping
        algorithm_config.scale_mapping = (
            "relu"  # The scaling of std_dev will be done in the model
        )
    else:
        model_config.probabilistic = False

    experiment = Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=cfg.seed,
        config=experiment_config,
        callbacks=[
            SndCallback(),
            # ExtremumSeekingController(
            #             control_group="agents",
            #             # control_group="adversary",
            #             initial_snd=0.0,
            #             dither_magnitude=0.2,
            #             dither_frequency_rad_s=1.0,
            #             integral_gain=-0.001,
            #             high_pass_cutoff_rad_s=1.0,
            #             low_pass_cutoff_rad_s=1.0,
            #             sampling_period=1.0
            # ),
            TrajectorySNDLoggerCallback(control_group="agents"),
            SNDVisualizerCallback(),
            # TrajectorySNDLoggerCallback(control_group="adversary"),
            # performaceLoggerCallback(control_group="agents", initial_snd=0.5),
            NormLoggerCallback(),
            ActionSpaceLoss(
                use_action_loss=cfg.use_action_loss, action_loss_lr=cfg.action_loss_lr
            ),
        ]
        + (
            [
                TagCurriculum(
                    cfg.simple_tag_freeze_policy_after_frames,
                    cfg.simple_tag_freeze_policy,
                )
            ]
            if task_name == "vmas/simple_tag"
            else []
        ),
    )
    return experiment


# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def hydra_experiment(cfg: DictConfig) -> None:
#     experiment = get_experiment(cfg=cfg)
#     experiment.run()


# if __name__ == "__main__":
#     hydra_experiment()

ABS_CONFIG_PATH = "/home/spatel/Desktop/ad2c/ControllingBehavioralDiversity/het_control/conf"
CONFIG_NAME = "balance_ippo_config"  # Make sure 'navigation_ippo.yaml' exists in the folder above!
SAVE_PATH = "/home/spatel/Desktop/ad2c/model_checkpoint/balance_ippo/"

save_interval = 1_200_000
desired_snd = -1.0
max_frame = 1_200_000

GlobalHydra.instance().clear()

sys.argv = [
    "dummy.py",
    f"model.desired_snd={desired_snd}",
    f"experiment.max_n_frames={max_frame}",
    f"experiment.checkpoint_interval={save_interval}",
    f"experiment.save_folder={SAVE_PATH}",
    # f"task.agents_with_same_goal=1",
    # f"task.n_agents=4",
]

# 3. Define the Hydra wrapper
@hydra.main(version_base=None, config_path=ABS_CONFIG_PATH, config_name=CONFIG_NAME)
def hydra_experiment(cfg: DictConfig) -> None:
    experiment = get_experiment(cfg=cfg)
    experiment.run()

# 4. Execute safely
if __name__ == "__main__":
    try:
        hydra_experiment()
    except SystemExit:
        print("Experiment finished successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")