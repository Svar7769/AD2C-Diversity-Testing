"""
Runner script for Navigation task with ESC control.
<<<<<<< HEAD
Supports dynamic CLI overrides for task, model, and experiment parameters.
"""
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any
from het_control.run import run_experiment


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_DIR = Path("/home/svarp/Desktop/Projects/ad2c - testEnv/AD2C-Diversity-Testing/")
ABS_CONFIG_PATH = str(BASE_DIR / "het_control/conf")
CONFIG_NAME = "navigation_ippo"
BASE_SAVE_PATH = Path("/home/svarp/Desktop/Projects/ad2c - testEnv/model_checkpoint/navigation_ippo")

# Training parameters
MAX_FRAMES = 6_000_000
CHECKPOINT_INTERVAL = 6_000_000
USE_ESC = True

# Task defaults
TASK_OVERRIDES = {
    "n_agents": 3,
    "agents_with_same_goal": 2
}

# ESC configuration (using new optimizer-based implementation)
ESC_PARAMS = {
    "control_group": "agents",
    "initial_snd": 1,
    "integrator_gain": -0.07,
    "dither_magnitude": 0.2,
    "dither_frequency": 1.0,
    "high_pass_cutoff": 1.0,
    "low_pass_cutoff": 1.0,
    "sampling_period": 1.0,
    "min_snd": 0.0,
    "max_snd": 3.0,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_value(value: str) -> Any:
    """Parse string value to appropriate type."""
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
        return float(value) if "." in value else int(value)
    return value


def parse_cli_args() -> tuple[int, int, Dict[str, Any], Dict[str, Any]]:
    """
    Parse command-line arguments and return updated configurations.
    
    Returns:
        Tuple of (max_frames, checkpoint_interval, task_overrides, esc_params)
    """
    max_frames = MAX_FRAMES
    checkpoint_interval = CHECKPOINT_INTERVAL
    task_overrides = TASK_OVERRIDES.copy()
    esc_params = ESC_PARAMS.copy()
    
    for arg in sys.argv[1:]:
        if "=" not in arg:
            continue
            
        key, value = arg.split("=", 1)
        value = parse_value(value)
        
        # Handle different override types
        if key.startswith("task."):
            task_key = key.split(".", 1)[1]
            task_overrides[task_key] = value
            print(f"🔧 Task Override: {task_key} = {value}")
            
        elif key.startswith("esc."):
            esc_key = key.split(".", 1)[1]
            esc_params[esc_key] = value
            print(f"🔧 ESC Override: {esc_key} = {value}")
            
        elif key == "model.desired_snd":
            esc_params["initial_snd"] = value
            print(f"🔧 ESC Initial SND: {value}")
            
        elif key == "max_frames":
            max_frames = value
            print(f"🔧 Max Frames: {value:,}")
            
        elif key == "checkpoint_interval":
            checkpoint_interval = value
            print(f"🔧 Checkpoint Interval: {value:,}")
            
        else:
            print(f"⚠️  Unknown parameter: {key}={value}")
    
    return max_frames, checkpoint_interval, task_overrides, esc_params


def create_save_path(base_path: Path, task_overrides: Dict, esc_params: Dict) -> Path:
    """Create a descriptive save path based on configuration."""
    goals = task_overrides.get("agents_with_same_goal", "std")
    snd = esc_params["initial_snd"]
    optimizer = esc_params.get("optimizer_type", "adam")
    
    save_path = base_path / f"snd_{snd}_goals_{goals}_{optimizer}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    return save_path


def create_esc_config(esc_params: Dict) -> str:
    """Create temporary ESC configuration file."""
    temp_config = "/tmp/esc_navigation.yaml"
    
    with open(temp_config, 'w') as f:
        yaml.dump({"esc_controller": esc_params}, f)
    
    return temp_config


def print_experiment_summary(
    save_path: Path,
    max_frames: int,
    checkpoint_interval: int,
    task_overrides: Dict,
    esc_params: Dict,
    use_esc: bool
):
    """Print experiment configuration summary."""
    print("\n" + "="*80)
    print("🚀 NAVIGATION EXPERIMENT")
    print("="*80)
    print(f"Save Path: {save_path}")
    print(f"Max Frames: {max_frames:,}")
    print(f"Checkpoint: Every {checkpoint_interval:,} frames")
    
    print(f"\n📋 Task Configuration:")
    for key, value in task_overrides.items():
        print(f"   {key}: {value}")
    
    if use_esc:
        print(f"\n🎛️  ESC Configuration:")
        print(f"   Initial SND: {esc_params['initial_snd']}")
        print(f"   Optimizer: {esc_params.get('optimizer_type', 'adam').upper()}")
        print(f"   Learning Rate: {esc_params.get('learning_rate', 0.01)}")
        print(f"   Dither: ±{esc_params['dither_magnitude']} @ {esc_params['dither_frequency']} rad/s")
        print(f"   Objective: {'MAXIMIZE reward' if esc_params.get('maximize', True) else 'MINIMIZE cost'}")
    else:
        print(f"\n🎛️  ESC: DISABLED")
    
    print("="*80 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    # Parse CLI arguments
    max_frames, checkpoint_interval, task_overrides, esc_params = parse_cli_args()
    
    # Create save path
    save_path = create_save_path(BASE_SAVE_PATH, task_overrides, esc_params)
    
    # Setup ESC configuration
    esc_config_path = create_esc_config(esc_params) if USE_ESC else None
    
    # Print summary
    print_experiment_summary(
        save_path=save_path,
        max_frames=max_frames,
        checkpoint_interval=checkpoint_interval,
        task_overrides=task_overrides,
        esc_params=esc_params,
        use_esc=USE_ESC
    )
    
    # Run experiment
    run_experiment(
        config_path=ABS_CONFIG_PATH,
        config_name=CONFIG_NAME,
        save_path=str(save_path),
        max_frames=max_frames,
        checkpoint_interval=checkpoint_interval,
        task_overrides=task_overrides,
        esc_config_path=esc_config_path,
=======
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
BASE_DIR = "/home/grad/doc/2027/spatel2/ad2c/AD2C-Diversity-Testing"

# Paths
ABS_CONFIG_PATH = f"{BASE_DIR}/het_control/conf"
CONFIG_NAME = "navigation_ippo"
SAVE_PATH = "/home/grad/doc/2027/spatel2/ad2c/model_checkpoint/navigation_ippo/"

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
    "integrator_gain": -1.00,
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
    Supports any Hydra-style override: category.parameter=value or seed=value
    
    Categories:
    - model.*: Model parameters
    - task.*: Task parameters
    - experiment.*: Experiment parameters
    - esc.*: ESC controller parameters
    - seed: Random seed
    
    Returns:
        tuple: (model_overrides, task_overrides, experiment_overrides, esc_overrides, seed)
    """
    model_overrides = {}
    task_overrides = {}
    experiment_overrides = {}
    esc_overrides = DEFAULT_ESC_OVERRIDES.copy()
    seed = None
    
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
        elif key == 'seed':
            seed = value
        else:
            print(f"⚠️  Warning: Unknown override category: {key}")
    
    return model_overrides, task_overrides, experiment_overrides, esc_overrides, seed


def run_navigation_experiment(
    model_overrides: dict = None,
    task_overrides: dict = None,
    experiment_overrides: dict = None,
    esc_overrides: dict = None,
    seed: int = None,
    use_esc: bool = USE_ESC
):
    """
    Run navigation experiment with specified parameters.
    
    Args:
        model_overrides: Dictionary of model parameter overrides
        task_overrides: Dictionary of task parameter overrides
        experiment_overrides: Dictionary of experiment parameter overrides
        esc_overrides: Dictionary of ESC controller parameter overrides
        seed: Random seed value
        use_esc: Whether to use ESC controller
    """
    # Set defaults if None
    model_overrides = model_overrides or {}
    task_overrides = task_overrides or {}
    experiment_overrides = experiment_overrides or {}
    esc_overrides = esc_overrides or DEFAULT_ESC_OVERRIDES.copy()
    
    # Add seed to experiment overrides if provided
    if seed is not None:
        experiment_overrides['seed'] = seed
    
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
            temp_esc_config = "/tmp/escontroller_navigation.yaml"
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
    print(f"🎯 Running Navigation Task with ESC")
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
        print(f"   Integrator gain: {esc_overrides.get('integrator_gain', -10.0)}")
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
    model_overrides, task_overrides, experiment_overrides, esc_overrides, seed = parse_all_overrides()
    
    # Print what was parsed
    print(f"\n{'='*50}")
    print(f"Parsed overrides:")
    if model_overrides:
        print(f"  Model: {model_overrides}")
    if task_overrides:
        print(f"  Task: {task_overrides}")
    if experiment_overrides:
        print(f"  Experiment: {experiment_overrides}")
    if seed is not None:
        print(f"  Seed: {seed}")
    if esc_overrides != DEFAULT_ESC_OVERRIDES:
        print(f"  ESC: {esc_overrides}")
    print(f"{'='*50}\n")
    
    # Run experiment with parsed parameters
    run_navigation_experiment(
        model_overrides=model_overrides,
        task_overrides=task_overrides,
        experiment_overrides=experiment_overrides,
        esc_overrides=esc_overrides,
        seed=seed,
>>>>>>> origin/device/lab
        use_esc=USE_ESC
    )


if __name__ == "__main__":
    main()