"""
Runner script for Balance task with ESC control.
Fully dynamic - supports any Hydra override from command line.
Path: /home/spatel/Desktop/ad2c/AD2C-Diversity-Testing/
"""
import sys
from het_control.run import run_experiment
import yaml
import os

# =============================================================================
# CONFIGURATION - Updated Paths for New System
# =============================================================================
# Base directory for the project
BASE_DIR = "/home/spatel/Desktop/ad2c/AD2C-Diversity-Testing"

# Paths
ABS_CONFIG_PATH = f"{BASE_DIR}/het_control/conf"
CONFIG_NAME = "balance_ippo_config"
SAVE_PATH = "/home/spatel/Desktop/ad2c/model_checkpoint/balance_ippo/"

# Default training parameters (can be overridden)
DEFAULT_MAX_FRAMES = 12_000_000
DEFAULT_CHECKPOINT_INTERVAL = 12_000_000

# ESC Controller Configuration
USE_ESC = True  # Set to False to disable ESC
ESC_CONFIG_FILE = f"{BASE_DIR}/het_control/conf/callback/escontroller.yaml"

# Default ESC overrides (can be overridden from command line)
DEFAULT_ESC_OVERRIDES = {
    "control_group": "agents",
    "dither_magnitude": 0.2,      
    "dither_frequency": 1.0,    
    "high_pass_cutoff": 0.05,
    "low_pass_cutoff": 0.5,
    "integrator_gain": -0.1,
    "sampling_period": 1.0,       
    "min_snd": 0.0,
    "max_snd": 3.0,
    "use_adaptive_gain": True,
    "use_action_loss": False,
    "action_loss_lr": 0.001,
}


def parse_all_overrides():
    """
    Parse all command line arguments dynamically.
    Supports any Hydra-style override: category.parameter=value
    
    Categories:
    - model.*: Model parameters
    - task.*: Task parameters (e.g., n_agents, package_mass)
    - experiment.*: Experiment parameters
    - esc.*: ESC controller parameters
    
    Returns:
        tuple: (model_overrides, task_overrides, experiment_overrides, esc_overrides)
    """
    model_overrides = {}
    task_overrides = {}
    experiment_overrides = {}
    esc_overrides = DEFAULT_ESC_OVERRIDES.copy()
    
    for arg in sys.argv[1:]:
        if '=' not in arg:
            continue
            
        key, value = arg.split('=', 1)
        
        # Try to convert value to appropriate type
        try:
            # Try float first
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # Try boolean
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            # Otherwise keep as string
        
        # Route to appropriate dictionary
        if key.startswith('model.'):
            param = key.replace('model.', '')
            model_overrides[param] = value
        elif key.startswith('task.'):
            param = key.replace('task.', '')
            task_overrides[param] = value
        elif key.startswith('experiment.'):
            param = key.replace('experiment.', '')
            experiment_overrides[param] = value
        elif key.startswith('esc.'):
            param = key.replace('esc.', '')
            esc_overrides[param] = value
        else:
            print(f"⚠️  Warning: Unknown override category: {key}")
    
    return model_overrides, task_overrides, experiment_overrides, esc_overrides


def run_balance_experiment(
    model_overrides: dict = None,
    task_overrides: dict = None,
    experiment_overrides: dict = None,
    esc_overrides: dict = None,
    use_esc: bool = USE_ESC
):
    """
    Run Balance experiment with specified parameters.
    
    Args:
        model_overrides: Dictionary of model parameter overrides
        task_overrides: Dictionary of task parameter overrides
        experiment_overrides: Dictionary of experiment parameter overrides
        esc_overrides: Dictionary of ESC controller parameter overrides
        use_esc: Whether to use ESC controller
    """
    # Set defaults if None
    model_overrides = model_overrides or {}
    task_overrides = task_overrides or {}
    experiment_overrides = experiment_overrides or {}
    esc_overrides = esc_overrides or DEFAULT_ESC_OVERRIDES.copy()
    
    # Extract key parameters
    desired_snd = model_overrides.get('desired_snd', 0.0)
    max_frames = experiment_overrides.get('max_n_frames', DEFAULT_MAX_FRAMES)
    checkpoint_interval = experiment_overrides.get('checkpoint_interval', DEFAULT_CHECKPOINT_INTERVAL)
    
    # Load ESC config and apply overrides
    temp_esc_config = None
    if use_esc:
        try:
            with open(ESC_CONFIG_FILE, 'r') as f:
                esc_config = yaml.safe_load(f)
            
            if 'esc_controller' not in esc_config:
                esc_config['esc_controller'] = {}
            
            # Set initial_snd from model override
            esc_config['esc_controller']['initial_snd'] = desired_snd
            
            # Apply all ESC overrides
            for key, value in esc_overrides.items():
                esc_config['esc_controller'][key] = value
            
            # Save to a task-specific temporary file
            temp_esc_config = "/tmp/escontroller_balance.yaml"
            with open(temp_esc_config, 'w') as f:
                yaml.dump(esc_config, f)
            
            esc_config_to_use = temp_esc_config
            
        except FileNotFoundError:
            print(f"⚠️  Warning: ESC config not found at {ESC_CONFIG_FILE}. Using defaults.")
            esc_config_to_use = ESC_CONFIG_FILE
    else:
        esc_config_to_use = ESC_CONFIG_FILE
    
    # Print configuration summary
    print(f"{'='*80}")
    print(f"🎯 Running Balance Task with ESC")
    print(f"{'='*80}")
    print(f"📊 Configuration:")
    print(f"   Max frames: {max_frames:,}")
    print(f"   Checkpoint interval: {checkpoint_interval:,}")
    
    if model_overrides:
        print(f"\n📝 Model overrides:")
        for key, value in model_overrides.items():
            print(f"   {key}: {value}")
    
    if task_overrides:
        print(f"\n📋 Task overrides:")
        for key, value in task_overrides.items():
            print(f"   {key}: {value}")
    
    if experiment_overrides:
        print(f"\n⚙️  Experiment overrides:")
        for key, value in experiment_overrides.items():
            print(f"   {key}: {value}")
    
    if use_esc:
        print(f"\n🎛️  ESC Controller:")
        print(f"   Control group: {esc_overrides.get('control_group', 'agents')}")
        print(f"   Initial SND: {desired_snd}")
        print(f"   Dither: ±{esc_overrides.get('dither_magnitude', 0.2)} @ {esc_overrides.get('dither_frequency', 1.0)} rad/s")
        print(f"   Integrator gain: {esc_overrides.get('integrator_gain', -0.1)}")
        print(f"   Filters: HPF={esc_overrides.get('high_pass_cutoff', 0.1)}, LPF={esc_overrides.get('low_pass_cutoff', 0.05)} rad/s")
        print(f"   SND bounds: [{esc_overrides.get('min_snd', 0.0)}, {esc_overrides.get('max_snd', 3.0)}]")
        print(f"   Adaptive gain: {esc_overrides.get('use_adaptive_gain', True)}")
    
    print(f"{'='*80}\n")
    
    # Execute via the reusable run_experiment function from run.py
    run_experiment(
        config_path=ABS_CONFIG_PATH,
        config_name=CONFIG_NAME,
        save_path=SAVE_PATH,
        max_frames=max_frames,
        checkpoint_interval=checkpoint_interval,
        desired_snd=desired_snd,
        task_overrides=task_overrides,
        esc_config_path=esc_config_to_use,
        use_esc=use_esc
    )


if __name__ == "__main__":
    # Parse all command-line arguments
    model_overrides, task_overrides, experiment_overrides, esc_overrides = parse_all_overrides()
    
    # Print what was parsed
    print(f"\n{'='*50}")
    print(f"Parsed overrides:")
    if model_overrides:
        print(f"  Model: {model_overrides}")
    if task_overrides:
        print(f"  Task: {task_overrides}")
    if experiment_overrides:
        print(f"  Experiment: {experiment_overrides}")
    if esc_overrides != DEFAULT_ESC_OVERRIDES:
        print(f"  ESC: {esc_overrides}")
    print(f"{'='*50}\n")
    
    # Run experiment with parsed parameters
    run_balance_experiment(
        model_overrides=model_overrides,
        task_overrides=task_overrides,
        experiment_overrides=experiment_overrides,
        esc_overrides=esc_overrides,
        use_esc=USE_ESC
    )