"""
Runner script for Simple Tag task with ESC control.
Example showing adversarial control group.
"""
from het_control.run import run_experiment
from pathlib import Path
import sys
import yaml

# =============================================================================
# CONFIGURATION - Update these for your system and experiment
# =============================================================================

# Paths
ABS_CONFIG_PATH = "/home/svarp/Desktop/Projects/ad2c - testEnv/AD2C-Diversity-Testing/het_control/conf"
CONFIG_NAME = "tag_ippo_config"
SAVE_PATH = "/home/svarp/Desktop/Projects/ad2c - testEnv/model_checkpoint/simple_tag_ippo/"

# Training parameters
MAX_FRAMES = 1_200_000
CHECKPOINT_INTERVAL = 1_200_000

# Initial SND
DESIRED_SND = 0.5

# Task-specific overrides
TASK_OVERRIDES = {
    "num_good_agents": 3,
    "num_adversaries": 2,
}

# ESC Controller
USE_ESC = True
ESC_CONFIG_FILE = "/home/svarp/Desktop/Projects/ad2c - testEnv/AD2C-Diversity-Testing/het_control/conf/callback/escontroller.yaml"

# ESC parameter overrides for adversarial control
ESC_OVERRIDES = {
    "control_group": "adversary",  # Override to control adversary group
    # You can override other params too:
    # "dither_magnitude": 0.3,
    # "max_snd": 4.0,
}

# =============================================================================
# RUN EXPERIMENT
# =============================================================================

if __name__ == "__main__":
    # Load ESC config and apply overrides
    temp_esc_config = None
    if USE_ESC and ESC_OVERRIDES:
        with open(ESC_CONFIG_FILE, 'r') as f:
            esc_config = yaml.safe_load(f)
        
        # Apply overrides
        for key, value in ESC_OVERRIDES.items():
            esc_config['esc_controller'][key] = value
            print(f"🔧 Overriding ESC parameter: {key} = {value}")
        
        # Save to temporary file
        temp_esc_config = "/tmp/escontroller_simple_tag.yaml"
        with open(temp_esc_config, 'w') as f:
            yaml.dump(esc_config, f)
        
        esc_config_to_use = temp_esc_config
    else:
        esc_config_to_use = ESC_CONFIG_FILE
    
    print(f"{'='*80}")
    print(f"🎯 Running Simple Tag Task (Adversarial ESC)")
    print(f"{'='*80}\n")
    
    run_experiment(
        config_path=ABS_CONFIG_PATH,
        config_name=CONFIG_NAME,
        save_path=SAVE_PATH,
        max_frames=MAX_FRAMES,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        desired_snd=DESIRED_SND,
        task_overrides=TASK_OVERRIDES,
        esc_config_path=esc_config_to_use,
        use_esc=USE_ESC
    )