"""
Runner script for Navigation task with ESC control.
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
MAX_FRAMES = 1_200_000
CHECKPOINT_INTERVAL = 1_200_000
USE_ESC = True

# Task defaults
TASK_OVERRIDES = {
    "n_agents": 3
}

# ESC configuration (using new optimizer-based implementation)
ESC_PARAMS = {
    "control_group": "agents",
    "initial_snd": 0.0,
    "dither_magnitude": 0.2,
    "dither_frequency": 1.0,
    "high_pass_cutoff": 0.5,
    "low_pass_cutoff": 0.1,
    "sampling_period": 1.0,
    "min_snd": 0.0,
    "max_snd": 3.0,
    "maximize": True,
    # Optimizer settings
    "optimizer_type": "adam",  # 'adam', 'rmsprop', or 'sgd'
    "learning_rate": 0.01,
    "betas": [0.9, 0.999],  # For Adam
    "alpha": 0.99,  # For RMSprop
    "momentum": 0.9,  # For SGD/RMSprop
    "use_lr_scheduler": False,
    "scheduler_type": "plateau",
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
        use_esc=USE_ESC
    )


if __name__ == "__main__":
    main()