"""
Reusable experiment runner for BenchMARL with ESC control.
Handles all common functionality across different tasks.
"""
import sys
import hydra
import yaml
from typing import Optional, Dict, Any
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra

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

from het_control.callbacks.callback import (
    SndCallback,
    NormLoggerCallback,
    ActionSpaceLoss,
    TagCurriculum
)
from het_control.callbacks.adaptiveEsc_callback import AdaptiveESCCallback
from het_control.callbacks.sndVisualCallback import SNDVisualizerCallback
from het_control.environments.vmas import render_callback
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpiricalConfig


# Constants
VMAS_TASKS = [
    "vmas/balance", "vmas/ball_passage", "vmas/ball_trajectory", "vmas/buzz_wire",
    "vmas/discovery", "vmas/dispersion", "vmas/football", "vmas/navigation",
    "vmas/reverse_transport", "vmas/sampling", "vmas/tag", "vmas/simple_tag"
]
ON_POLICY_ALGORITHMS = (MappoConfig, IppoConfig, MasacConfig, IsacConfig)


def setup(task_name: str) -> None:
    """Register custom models and setup task-specific configurations."""
    benchmarl.models.model_config_registry["hetcontrolmlpempirical"] = HetControlMlpEmpiricalConfig
    
    if task_name in VMAS_TASKS:
        VmasTask.render_callback = render_callback


def configure_model_for_algorithm(model_config, algorithm_config):
    """Configure model based on algorithm type."""
    if isinstance(algorithm_config, ON_POLICY_ALGORITHMS):
        model_config.probabilistic = True
        model_config.scale_mapping = algorithm_config.scale_mapping
        algorithm_config.scale_mapping = "relu"
    else:
        model_config.probabilistic = False


def create_esc_callback(esc_config: Dict[str, Any]) -> AdaptiveESCCallback:
    """Create ESC callback from configuration dictionary."""
    return AdaptiveESCCallback(
                control_group=esc_config.get("control_group", "agents"),
                initial_snd=esc_config.get("initial_snd", 0.0),
                dither_magnitude=esc_config.get("dither_magnitude", 0.2),
                dither_frequency_rad_s=esc_config.get("dither_frequency", 1.0),
                integrator_gain=esc_config.get("integrator_gain", -0.001),
                high_pass_cutoff_rad_s=esc_config.get("high_pass_cutoff", 0.5),
                low_pass_cutoff_rad_s=esc_config.get("low_pass_cutoff", 0.1),
                use_adaptive_gain=esc_config.get("use_adaptive_gain", True),
                sampling_period=esc_config.get("sampling_period", 1.0),
                min_snd=esc_config.get("min_snd", 0.0),
                max_snd=esc_config.get("max_snd", 3.0)
            )


def get_experiment(cfg: DictConfig, esc_config: Optional[Dict[str, Any]] = None) -> Experiment:
    """
    Create and configure the BenchMARL experiment.
    
    Args:
        cfg: Hydra configuration dictionary
        esc_config: ESC controller configuration (optional)
        
    Returns:
        Configured Experiment object
    """
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm
    
    setup(task_name)
    
    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))
    
    # Load all configurations
    algorithm_config = load_algorithm_config_from_hydra(cfg.algorithm)
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)
    task_config = load_task_config_from_hydra(cfg.task, task_name)
    critic_model_config = load_model_config_from_hydra(cfg.critic_model)
    model_config = load_model_config_from_hydra(cfg.model)
    
    # Configure model for algorithm
    configure_model_for_algorithm(model_config, algorithm_config)
    
    # Initialize base callbacks
    callbacks = [
        SndCallback(),
        NormLoggerCallback(),
        ActionSpaceLoss(
            use_action_loss=esc_config.get("use_action_loss", True) if esc_config else cfg.get("use_action_loss", True),
            action_loss_lr=esc_config.get("action_loss_lr", 0.001) if esc_config else cfg.get("action_loss_lr", 0.001)
        ),
        SNDVisualizerCallback()
    ]
    
    # Add ESC callback if config provided
    if esc_config is not None:
        # Set initial desired_snd in model config
        if hasattr(model_config, 'desired_snd'):
            model_config.desired_snd = esc_config.get("initial_snd", 1.0)
        
        callbacks.insert(2, create_esc_callback(esc_config))  # Insert after NormLogger
    
    # Add task-specific callbacks
    if task_name == "vmas/simple_tag":
        freeze_after = esc_config.get("simple_tag_freeze_policy_after_frames", 1_000_000) if esc_config else cfg.get("simple_tag_freeze_policy_after_frames", 1_000_000)
        freeze_policy = esc_config.get("simple_tag_freeze_policy", False) if esc_config else cfg.get("simple_tag_freeze_policy", False)
        callbacks.append(TagCurriculum(freeze_after, freeze_policy))
    
    return Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=cfg.seed,
        config=experiment_config,
        callbacks=callbacks
    )


def load_esc_config(config_path: str) -> Dict[str, Any]:
    """Load ESC configuration from YAML file."""
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
        return full_config.get('esc_controller', {})


def print_experiment_header(
    config_path: str,
    config_name: str,
    save_path: str,
    max_frames: int,
    checkpoint_interval: int,
    desired_snd: float,
    esc_config: Optional[Dict[str, Any]],
    task_overrides: Optional[Dict[str, Any]]
) -> None:
    """Print experiment configuration summary."""
    print("\n" + "="*80)
    print("Starting Experiment")
    print("="*80)
    print(f"Config: {config_path}/{config_name}.yaml")
    print(f"Save path: {save_path}")
    print(f"Max frames: {max_frames:,}")
    print(f"Checkpoint interval: {checkpoint_interval:,}")
    print(f"Desired SND: {desired_snd}")
    
    if esc_config:
        print(f"\n🎛️  ESC Controller: ENABLED")
        print(f"   Mode: {esc_config.get('optimizer_type', 'adam').upper()} optimizer")
        print(f"   Control group: {esc_config.get('control_group', 'agents')}")
        print(f"   Initial SND: {esc_config.get('initial_snd', 1.0)}")
        print(f"   Objective: {'MAXIMIZE reward' if esc_config.get('maximize', True) else 'MINIMIZE cost'}")
        print(f"   Learning rate: {esc_config.get('learning_rate', 0.01)}")
        print(f"   Dither: ±{esc_config.get('dither_magnitude', 0.1)} @ {esc_config.get('dither_frequency', 0.5)} rad/s")
        print(f"   HPF/LPF cutoff: {esc_config.get('high_pass_cutoff', 0.1)}/{esc_config.get('low_pass_cutoff', 0.05)} rad/s")
        print(f"   SND bounds: [{esc_config.get('min_snd', 0.0)}, {esc_config.get('max_snd', 3.0)}]")
        if esc_config.get('use_lr_scheduler', False):
            print(f"   LR Scheduler: {esc_config.get('scheduler_type', 'plateau')}")
    else:
        print(f"\n🎛️  ESC Controller: DISABLED")
    
    if task_overrides:
        print(f"\nTask overrides:")
        for param, value in task_overrides.items():
            print(f"  {param}: {value}")
    
    print("="*80 + "\n")


def build_hydra_args(
    max_frames: int,
    checkpoint_interval: int,
    save_path: str,
    desired_snd: float,
    task_overrides: Optional[Dict[str, Any]] = None
) -> list:
    """Build Hydra command-line arguments."""
    args = [
        "dummy.py",
        f"experiment.max_n_frames={max_frames}",
        f"experiment.checkpoint_interval={checkpoint_interval}",
        f"experiment.save_folder={save_path}",
        f"model.desired_snd={desired_snd}"
    ]
    
    if task_overrides:
        args.extend([f"task.{k}={v}" for k, v in task_overrides.items()])
    
    return args


def run_experiment(
    config_path: str,
    config_name: str,
    save_path: str,
    max_frames: int,
    checkpoint_interval: int,
    desired_snd: float = 1.0,
    task_overrides: Optional[Dict[str, Any]] = None,
    esc_config_path: Optional[str] = None,
    use_esc: bool = True
):
    """
    Run the experiment with specified configuration.
    
    Args:
        config_path: Absolute path to Hydra config directory
        config_name: Name of the Hydra config file (without .yaml)
        save_path: Directory to save checkpoints
        max_frames: Total training frames
        checkpoint_interval: Save checkpoint every N frames
        desired_snd: Initial desired SND value
        task_overrides: Dictionary of task parameter overrides
        esc_config_path: Path to ESC configuration YAML
        use_esc: Whether to use ESC controller
    """
    # Load ESC configuration
    esc_config = None
    if use_esc and esc_config_path:
        esc_config = load_esc_config(esc_config_path)
        
        print("\n" + "="*80)
        print("📄 Loaded ESC Configuration:")
        print("="*80)
        for key, value in esc_config.items():
            print(f"  {key}: {value}")
        print("="*80 + "\n")
        
        # Use ESC's initial_snd if available
        desired_snd = esc_config.get('initial_snd', desired_snd)
    
    # Clear existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Build Hydra arguments
    sys.argv = build_hydra_args(
        max_frames=max_frames,
        checkpoint_interval=checkpoint_interval,
        save_path=save_path,
        desired_snd=desired_snd,
        task_overrides=task_overrides
    )
    
    # Print experiment summary
    print_experiment_header(
        config_path=config_path,
        config_name=config_name,
        save_path=save_path,
        max_frames=max_frames,
        checkpoint_interval=checkpoint_interval,
        desired_snd=desired_snd,
        esc_config=esc_config if use_esc else None,
        task_overrides=task_overrides
    )
    
    @hydra.main(
        version_base=None,
        config_path=config_path,
        config_name=config_name
    )
    def hydra_experiment(cfg: DictConfig) -> None:
        experiment = get_experiment(cfg=cfg, esc_config=esc_config if use_esc else None)
        experiment.run()
    
    # Execute experiment
    try:
        hydra_experiment()
        print("\n" + "="*80)
        print("✅ Experiment finished successfully!")
        print("="*80 + "\n")
    except SystemExit:
        print("\n" + "="*80)
        print("⚠️  Experiment terminated.")
        print("="*80 + "\n")
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ ERROR: {e}")
        print("="*80 + "\n")
        raise