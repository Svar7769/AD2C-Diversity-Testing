"""
Reusable experiment runner for BenchMARL with ESC control.
Handles all common functionality across different tasks.
"""
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra
import yaml
from typing import Optional, Dict, Any

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
from het_control.callbacks.esc_callback import ESCCallback  # Changed from AdaptiveESCCallback
from het_control.callbacks.sndVisualCallback import SNDVisualizerCallback
from het_control.environments.vmas import render_callback
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpiricalConfig


def setup(task_name: str) -> None:
    """Register custom models and setup task-specific configurations."""
    benchmarl.models.model_config_registry.update({
        "hetcontrolmlpempirical": HetControlMlpEmpiricalConfig,
    })
    
    # Task-specific render callbacks
    vmas_tasks = [
        "vmas/balance", "vmas/ball_passage", "vmas/ball_trajectory", "vmas/buzz_wire",
        "vmas/discovery", "vmas/dispersion", "vmas/football", "vmas/navigation", 
        "vmas/reverse_transport", "vmas/sampling", "vmas/tag"
    ]
    if task_name in vmas_tasks:
        VmasTask.render_callback = render_callback


def get_experiment(cfg: DictConfig, esc_config: Optional[Dict[str, Any]] = None) -> Experiment:
    """
    Create and configure the BenchMARL experiment with all callbacks.
    
    Args:
        cfg: Hydra configuration dictionary
        esc_config: ESC controller configuration dictionary (optional)
        
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
    
    # Load configurations
    algorithm_config = load_algorithm_config_from_hydra(cfg.algorithm)
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)
    task_config = load_task_config_from_hydra(cfg.task, task_name)
    critic_model_config = load_model_config_from_hydra(cfg.critic_model)
    model_config = load_model_config_from_hydra(cfg.model)
    
    # Configure probabilistic policies for on-policy algorithms
    if isinstance(algorithm_config, (MappoConfig, IppoConfig, MasacConfig, IsacConfig)):
        model_config.probabilistic = True
        model_config.scale_mapping = algorithm_config.scale_mapping
        algorithm_config.scale_mapping = "relu"
    else:
        model_config.probabilistic = False
    
    # Initialize base callbacks (always included)
    callbacks = [
        SndCallback(),
        NormLoggerCallback(),
    ]
    
    # Add ESC-related callbacks if ESC config is provided
    if esc_config is not None:
        control_group = esc_config.get("control_group", "agents")
        
        # Set initial desired_snd in model config
        if hasattr(model_config, 'desired_snd'):
            model_config.desired_snd = esc_config.get("initial_snd", 0.0)
        
        # Add ESC controller
        callbacks.append(
            ESCCallback(
                control_group=control_group,
                initial_snd=esc_config.get("initial_snd", 0.0),
                dither_magnitude=esc_config.get("dither_magnitude", 0.1),
                dither_frequency_rad_s=esc_config.get("dither_frequency", 0.5),
                integrator_gain=esc_config.get("integrator_gain", -0.01),
                high_pass_cutoff_rad_s=esc_config.get("high_pass_cutoff", 0.1),
                low_pass_cutoff_rad_s=esc_config.get("low_pass_cutoff", 0.05),
                use_adaptive_gain=esc_config.get("use_adaptive_gain", True),
                sampling_period=esc_config.get("sampling_period", 1.0),
                min_snd=esc_config.get("min_snd", 0.0),
                max_snd=esc_config.get("max_snd", 3.0),
            )
        )
        
        # Add action space loss
        callbacks.append(
            ActionSpaceLoss(
                use_action_loss=esc_config.get("use_action_loss", False),
                action_loss_lr=esc_config.get("action_loss_lr", 0.001)
            )
        )
    else:
        # No ESC - use action loss from cfg if available
        callbacks.append(
            ActionSpaceLoss(
                use_action_loss=cfg.get("use_action_loss", False),
                action_loss_lr=cfg.get("action_loss_lr", 0.001)
            )
        )
    
    # Always add SND visualizer
    callbacks.append(SNDVisualizerCallback())
    
    # Add task-specific callbacks
    if task_name == "vmas/simple_tag":
        freeze_after = esc_config.get("simple_tag_freeze_policy_after_frames", 1_000_000) if esc_config else cfg.get("simple_tag_freeze_policy_after_frames", 1_000_000)
        freeze_policy = esc_config.get("simple_tag_freeze_policy", False) if esc_config else cfg.get("simple_tag_freeze_policy", False)
        callbacks.append(TagCurriculum(freeze_after, freeze_policy))
    
    # Create experiment
    experiment = Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=cfg.seed,
        config=experiment_config,
        callbacks=callbacks
    )
    
    return experiment


def load_esc_config(config_path: str) -> Dict[str, Any]:
    """
    Load ESC controller configuration from YAML file.
    Extracts the esc_controller section directly.
    """
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
        return full_config.get('esc_controller', {})


def run_experiment(
    config_path: str,
    config_name: str,
    save_path: str,
    max_frames: int,
    checkpoint_interval: int,
    desired_snd: float = 0.0,
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
        desired_snd: Initial desired SND value (for model initialization)
        task_overrides: Dictionary of task parameter overrides
        esc_config_path: Path to ESC configuration YAML (optional)
        use_esc: Whether to use ESC controller
    """
    # Load ESC configuration if provided
    esc_config = None
    if use_esc and esc_config_path is not None:
        esc_config = load_esc_config(esc_config_path)
        
        # Print loaded configuration
        print("\n" + "="*80)
        print("📄 Loaded ESC Configuration:")
        print("="*80)
        for key, value in esc_config.items():
            print(f"  {key}: {value}")
        print("="*80 + "\n")
        
        # Use ESC's initial_snd if available
        if 'initial_snd' in esc_config:
            desired_snd = esc_config['initial_snd']
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Build command-line arguments for Hydra
    sys.argv = ["dummy.py"]
    
    # Add experiment configuration
    sys.argv.extend([
        f"experiment.max_n_frames={max_frames}",
        f"experiment.checkpoint_interval={checkpoint_interval}",
        f"experiment.save_folder={save_path}",
        f"model.desired_snd={desired_snd}",
    ])
    
    # Add ESC parameters if using ESC
    if use_esc and esc_config is not None:
        # Core ESC parameters (add with + prefix as they don't exist in base config)
        esc_params = {
            "initial_snd": esc_config.get('initial_snd', 0.0),
            "esc_dither_magnitude": esc_config.get('dither_magnitude', 0.1),
            "esc_dither_frequency": esc_config.get('dither_frequency', 0.5),
            "esc_integrator_gain": esc_config.get('integrator_gain', -0.01),
            "esc_high_pass_cutoff": esc_config.get('high_pass_cutoff', 0.1),
            "esc_low_pass_cutoff": esc_config.get('low_pass_cutoff', 0.05),
            "esc_use_adaptive_gain": esc_config.get('use_adaptive_gain', True),
            "esc_sampling_period": esc_config.get('sampling_period', 1.0),
            "esc_min_snd": esc_config.get('min_snd', 0.0),
            "esc_max_snd": esc_config.get('max_snd', 3.0),
        }
        
        for param, value in esc_params.items():
            sys.argv.append(f"+{param}={value}")
        
        # Override existing parameters (without + prefix)
        sys.argv.extend([
            f"use_action_loss={esc_config.get('use_action_loss', False)}",
            f"action_loss_lr={esc_config.get('action_loss_lr', 0.001)}",
        ])
    
    # Add task overrides if provided
    if task_overrides:
        for param, value in task_overrides.items():
            sys.argv.append(f"task.{param}={value}")
    
    # Print configuration summary
    print("\n" + "="*80)
    print("🚀 Starting Experiment")
    print("="*80)
    print(f"Config: {config_path}/{config_name}")
    print(f"Save path: {save_path}")
    print(f"Max frames: {max_frames:,}")
    print(f"Checkpoint interval: {checkpoint_interval:,}")
    print(f"Initial SND: {desired_snd}")
    
    if use_esc and esc_config:
        print(f"\n🎛️  ESC Controller: ENABLED")
        
        # Determine mode
        if esc_config.get('use_adaptive_gain', True):
            mode_desc = "ESC with Adaptive Gain"
        else:
            mode_desc = "Classical ESC"
        
        print(f"   Mode: {mode_desc}")
        print(f"   Control group: {esc_config.get('control_group', 'agents')}")
        print(f"   Initial SND: {esc_config.get('initial_snd', 0.0)}")
        print(f"   Dither: ±{esc_config.get('dither_magnitude', 0.1)} @ {esc_config.get('dither_frequency', 0.5)} rad/s")
        print(f"   Integrator gain: {esc_config.get('integrator_gain', -0.01)}")
        print(f"   Filters: HPF={esc_config.get('high_pass_cutoff', 0.1)}, LPF={esc_config.get('low_pass_cutoff', 0.05)} rad/s")
        print(f"   SND bounds: [{esc_config.get('min_snd', 0.0)}, {esc_config.get('max_snd', 3.0)}]")
    else:
        print(f"\n🎛️  ESC Controller: DISABLED")
    
    if task_overrides:
        print(f"\n📋 Task overrides:")
        for param, value in task_overrides.items():
            print(f"   {param}: {value}")
    
    print("="*80 + "\n")
    
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
        print("✅ Experiment completed successfully!")
        print("="*80 + "\n")
    except SystemExit:
        print("\n" + "="*80)
        print("⚠️  Experiment terminated")
        print("="*80 + "\n")
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ ERROR: {e}")
        print("="*80 + "\n")
        raise