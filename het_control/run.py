"""
Reusable experiment runner for BenchMARL with ESC control.
Handles all common functionality across different tasks.
Updated with logger monkey patch for hierarchical models.
"""
import sys
import hydra
from pathlib import Path
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
from het_control.callbacks.esc_callback import ESCCallback
from het_control.callbacks.sndESLogger import TrajectorySNDLoggerCallback
from het_control.callbacks.sndVisualCallback import SNDVisualizerCallback
from het_control.callbacks.subteam_assignment_logger import SubteamAssignmentLoggerCallback
from het_control.environments.vmas import render_callback
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpiricalConfig
from het_control.models.het_control_mlp_hierarchical import HetControlMlpHierarchicalConfig 


def patch_benchmarl_logger():
    """
    Apply monkey patch to fix batch size mismatch issues in logger.
    This is necessary for hierarchical models with minibatches of different sizes.
    """
    import torch
    from tensordict import TensorDictBase
    from tensordict._lazy import LazyStackedTensorDict
    from benchmarl.experiment.logger import Logger
    
    def patched_log_training(self, group: str, training_td: TensorDictBase, step: int):
        """
        Patched version that handles LazyStackedTensorDict with mismatched batch sizes.
        
        Instead of using .get() which triggers stacking, we manually extract from
        the underlying tensordicts and compute statistics.
        """
        if not len(self.loggers):
            return
        
        to_log = {}
        
        # Check if this is a LazyStackedTensorDict (which causes issues with mismatched batches)
        if isinstance(training_td, LazyStackedTensorDict):
            # Get the list of underlying tensordicts
            try:
                # Access the internal list of tensordicts
                tensordicts = training_td.tensordicts
            except AttributeError:
                # Fallback: try to get keys and process individually
                try:
                    keys = list(training_td.keys())
                except RuntimeError:
                    # Can't even get keys, skip logging
                    return
                
                # Try to extract each key without triggering stacking
                for key in keys:
                    try:
                        # Use get_nestedtensor which doesn't stack
                        nested_value = training_td.get_nestedtensor(key)
                        # nested_value is a NestedTensor - compute mean across all elements
                        if hasattr(nested_value, 'values'):
                            # For NestedTensor
                            values = [v.mean().item() for v in nested_value.values()]
                            to_log[f"train/{group}/{key}"] = sum(values) / len(values)
                        else:
                            # Regular tensor
                            to_log[f"train/{group}/{key}"] = nested_value.mean().item()
                    except (RuntimeError, AttributeError, KeyError):
                        # Skip keys that can't be accessed
                        continue
                
                if to_log:
                    self.log(to_log, step=step)
                return
            
            # If we successfully got tensordicts, process them manually
            all_values = {}
            for td in tensordicts:
                try:
                    for key in td.keys():
                        value = td.get(key)
                        if isinstance(value, torch.Tensor):
                            if key not in all_values:
                                all_values[key] = []
                            all_values[key].append(value.mean().item())
                except (RuntimeError, AttributeError):
                    continue
            
            # Average across all tensordicts
            for key, values in all_values.items():
                if values:
                    to_log[f"train/{group}/{key}"] = sum(values) / len(values)
        
        else:
            # Not a LazyStackedTensorDict - use normal processing
            try:
                keys = list(training_td.keys())
            except RuntimeError:
                return
            
            for key in keys:
                try:
                    value = training_td.get(key)
                    if isinstance(value, torch.Tensor):
                        to_log[f"train/{group}/{key}"] = value.mean().item()
                    else:
                        try:
                            to_log[f"train/{group}/{key}"] = value.mean().item()
                        except (RuntimeError, AttributeError):
                            pass
                except RuntimeError:
                    continue
        
        if to_log:
            self.log(to_log, step=step)
    
    Logger.log_training = patched_log_training
    print("✅ Applied logger monkey patch for hierarchical model compatibility")


def setup(task_name: str) -> None:
    """Register custom models and setup task-specific configurations."""
    benchmarl.models.model_config_registry.update({
        "hetcontrolmlpempirical": HetControlMlpEmpiricalConfig,
        "hetcontrolmlphierarchical": HetControlMlpHierarchicalConfig,
    })
    
    # Task-specific render callbacks
    if task_name in [
        "vmas/balance", "vmas/ball_passage", "vmas/ball_trajectory", 
        "vmas/buzz_wire", "vmas/discovery", "vmas/dispersion", 
        "vmas/football", "vmas/navigation", "vmas/reverse_transport",
        "vmas/sampling", "vmas/tag"
    ]:
        VmasTask.render_callback = render_callback


def get_experiment(
    cfg: DictConfig, 
    esc_config: Optional[Dict[str, Any]] = None,
    use_hierarchical: bool = False
) -> Experiment:
    """
    Create and configure the BenchMARL experiment with all callbacks.
    
    Args:
        cfg: Hydra configuration dictionary
        esc_config: ESC controller configuration dictionary (optional)
        use_hierarchical: Whether using hierarchical model (for callback selection)
        
    Returns:
        Configured Experiment object
    """
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm
    
    setup(task_name)
    
    # Apply logger patch for hierarchical models
    if use_hierarchical:
        patch_benchmarl_logger()
    
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
        )
        
        # Add ESC trajectory logger
        callbacks.append(TrajectorySNDLoggerCallback(control_group=control_group))
        
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
    
    # Add subteam assignment logger for hierarchical models
    if use_hierarchical:
        control_group = esc_config.get("control_group", "agents") if esc_config else "agents"
        log_interval = esc_config.get("subteam_log_interval", 10) if esc_config else 10
        callbacks.append(
            SubteamAssignmentLoggerCallback(
                control_group=control_group,
                log_interval=log_interval
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
        # Extract the nested esc_controller section
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
    use_esc: bool = True,
    use_hierarchical: bool = False
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
        use_hierarchical: Whether to use hierarchical model
    """
    # Load ESC configuration if provided
    esc_config = None
    if use_esc and esc_config_path is not None:
        esc_config = load_esc_config(esc_config_path)
        
        # Print what was loaded for debugging
        print("\n" + "="*80)
        print("📄 Loaded ESC Configuration from file:")
        print("="*80)
        for key, value in esc_config.items():
            print(f"  {key}: {value}")
        print("="*80 + "\n")
        
        # Use ESC's initial_snd if not explicitly overridden
        if 'initial_snd' in esc_config:
            desired_snd = esc_config['initial_snd']
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Build command-line arguments for Hydra
    sys.argv = ["dummy.py"]
    
    # Select model type
    if use_hierarchical:
        sys.argv.append("model=hetcontrolmlphierarchical")
    else:
        sys.argv.append("model=hetcontrolmlpempirical")
    
    # Add experiment configuration
    sys.argv.extend([
        f"experiment.max_n_frames={max_frames}",
        f"experiment.checkpoint_interval={checkpoint_interval}",
        f"experiment.save_folder={save_path}",
    ])
    
    # Set model.desired_snd (required for model initialization)
    sys.argv.append(f"model.desired_snd={desired_snd}")
    
    # Add hierarchical-specific parameters
    if use_hierarchical:
        n_subteams = esc_config.get("n_subteams", 3) if esc_config else 3
        subteam_tau = esc_config.get("subteam_tau", 0.1) if esc_config else 0.1
        use_hard = esc_config.get('use_hard_assignment', False) if esc_config else False
        normalize_w = esc_config.get('normalize_weights', False) if esc_config else False
        clip_w = esc_config.get('clip_weights', True) if esc_config else True
        
        sys.argv.append(f"model.n_subteams={n_subteams}")
        sys.argv.append(f"model.subteam_tau={subteam_tau}")
        sys.argv.append(f"model.use_hard_assignment={str(use_hard).lower()}")
        sys.argv.append(f"model.normalize_weights={str(normalize_w).lower()}")
        sys.argv.append(f"model.clip_weights={str(clip_w).lower()}")
        
        # Initial Weights
        shared_w = esc_config.get('shared_weight_init', 1.0) if esc_config else 1.0
        subteam_w = esc_config.get('subteam_weight_init', 0.5) if esc_config else 0.5
        agent_w = esc_config.get('agent_weight_init', 0.25) if esc_config else 0.25
        
        sys.argv.append(f"model.shared_weight_init={shared_w}")
        sys.argv.append(f"model.subteam_weight_init={subteam_w}")
        sys.argv.append(f"model.agent_weight_init={agent_w}")
    
    # Add ESC parameters if using ESC
    if use_esc and esc_config is not None:
        # Add new ESC parameters with + prefix (directly from loaded config)
        esc_params_to_add = {
            "initial_snd": esc_config.get('initial_snd', 0.0),
            "dither_magnitude": esc_config.get('dither_magnitude', 0.2),
            "dither_frequency": esc_config.get('dither_frequency', 1.0),
            "integrator_gain": esc_config.get('integrator_gain', -0.001),
            "high_pass_cutoff": esc_config.get('high_pass_cutoff', 0.5),
            "low_pass_cutoff": esc_config.get('low_pass_cutoff', 0.1),
            "use_adaptive_gain": str(esc_config.get('use_adaptive_gain', True)).lower(),
            "sampling_period": esc_config.get('sampling_period', 1.0),
            "min_snd": esc_config.get('min_snd', 0.0),
            "max_snd": esc_config.get('max_snd', 3.0),
        }
        
        for param, value in esc_params_to_add.items():
            sys.argv.append(f"+{param}={value}")
        
        # Override existing parameters
        use_action_loss = esc_config.get('use_action_loss', False)
        esc_override_params = {
            "use_action_loss": str(use_action_loss).lower(),
            "action_loss_lr": esc_config.get('action_loss_lr', 0.001),
        }
        
        for param, value in esc_override_params.items():
            sys.argv.append(f"{param}={value}")
    
    # Add task overrides if provided
    if task_overrides:
        for param, value in task_overrides.items():
            sys.argv.append(f"task.{param}={value}")
    
    # Print configuration summary
    print("\n" + "="*80)
    print("Starting Experiment")
    print("="*80)
    print(f"Config path: {config_path}")
    print(f"Config name: {config_name}")
    print(f"Save path: {save_path}")
    print(f"Max frames: {max_frames:,}")
    print(f"Checkpoint interval: {checkpoint_interval:,}")
    print(f"Desired SND: {desired_snd}")
    print(f"Model type: {'Hierarchical' if use_hierarchical else 'Empirical'}")
    
    if use_hierarchical:
        print(f"\n🏗️  Hierarchical Model Configuration:")
        print(f"  Subteams: {n_subteams}")
        print(f"  Subteam tau: {subteam_tau}")
        print(f"  Hard assignment: {use_hard}")
        print(f"  Normalize weights: {normalize_w}")
        print(f"  Clip weights: {clip_w}")
        print(f"  Weights (shared/subteam/agent): {shared_w}/{subteam_w}/{agent_w}")
    
    if use_esc and esc_config:
        print(f"\n🎛️  ESC Controller: ENABLED")
        print(f"Control group: {esc_config.get('control_group', 'agents')}")
        print(f"  Dither: ±{esc_config.get('dither_magnitude', 0.2)} @ {esc_config.get('dither_frequency', 1.0)} rad/s")
        print(f"  Integrator gain: {esc_config.get('integrator_gain', -0.001)}")
        print(f"  HPF cutoff: {esc_config.get('high_pass_cutoff', 0.5)} rad/s")
        print(f"  LPF cutoff: {esc_config.get('low_pass_cutoff', 0.1)} rad/s")
        print(f"  Adaptive gain: {esc_config.get('use_adaptive_gain', True)}")
        print(f"  SND bounds: [{esc_config.get('min_snd', 0.0)}, {esc_config.get('max_snd', 3.0)}]")
    else:
        print(f"\n🎛️  ESC Controller: DISABLED")
    
    if task_overrides:
        print(f"\nTask overrides:")
        for param, value in task_overrides.items():
            print(f"  {param}: {value}")
    
    print("="*80 + "\n")
    
    @hydra.main(
        version_base=None,
        config_path=config_path,
        config_name=config_name
    )
    def hydra_experiment(cfg: DictConfig) -> None:
        experiment = get_experiment(
            cfg=cfg, 
            esc_config=esc_config if use_esc else None,
            use_hierarchical=use_hierarchical
        )
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
        print(f"❌ ERROR: An error occurred: {e}")
        print("="*80 + "\n")
        raise