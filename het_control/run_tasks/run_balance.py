"""
Runner script for Balance task with ESC control.
All task-specific parameters are defined here.
"""
from het_control.run import run_experiment
from pathlib import Path
import sys

# =============================================================================
# CONFIGURATION - Update these for your system and experiment
# =============================================================================

# Paths
ABS_CONFIG_PATH = "/home/svarp/Desktop/Projects/ad2c - testEnv/AD2C-Diversity-Testing/het_control/conf"
CONFIG_NAME = "balance_ippo_config"
SAVE_PATH = "/home/svarp/Desktop/Projects/ad2c - testEnv/model_checkpoint/balance_ippo/"

# Training parameters
MAX_FRAMES = 1_200_000
CHECKPOINT_INTERVAL = 1_200_000

# Initial SND (will be overridden by ESC config if provided)
DESIRED_SND = 0.0

# Task-specific overrides (optional)
TASK_OVERRIDES = {
    # "n_agents": 4,
    # "agents_with_same_goal": 1,
}

# ESC Controller
USE_ESC = True  # Set to False to disable ESC
ESC_CONFIG_FILE = "/home/svarp/Desktop/Projects/ad2c - testEnv/AD2C-Diversity-Testing/het_control/conf/callback/escontroller.yaml"

# =============================================================================
# RUN EXPERIMENT
# =============================================================================

if __name__ == "__main__":
    # Get ESC config path
    script_dir = Path(__file__).parent
    
    if USE_ESC:
        # Check if ESC_CONFIG_FILE is absolute or relative
        esc_config_path = Path(ESC_CONFIG_FILE)
        if not esc_config_path.is_absolute():
            esc_config_path = script_dir / ESC_CONFIG_FILE
        
        # Verify file exists
        if not esc_config_path.exists():
            print(f"\n{'='*80}")
            print(f"❌ ERROR: ESC config file not found!")
            print(f"{'='*80}")
            print(f"Looking for: {esc_config_path}")
            print(f"\nOptions:")
            print(f"1. Create the file at the location above")
            print(f"2. Update ESC_CONFIG_FILE path in this script")
            print(f"3. Set USE_ESC = False to run without ESC")
            print(f"{'='*80}\n")
            sys.exit(1)
        
        esc_config_path = str(esc_config_path)
    else:
        esc_config_path = None
    
    print(f"{'='*80}")
    print(f"🎯 Running Balance Task")
    print(f"{'='*80}\n")
    
    run_experiment(
        config_path=ABS_CONFIG_PATH,
        config_name=CONFIG_NAME,
        save_path=SAVE_PATH,
        max_frames=MAX_FRAMES,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        desired_snd=DESIRED_SND,
        task_overrides=TASK_OVERRIDES,
        esc_config_path=esc_config_path,
        use_esc=USE_ESC
    )