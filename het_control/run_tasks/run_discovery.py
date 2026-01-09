"""
Runner script for Discovery task with ESC control.
Updated for the new system path: /home/svarp/Desktop/Projects/ad2c - testEnv/
"""
from het_control.run import run_experiment
import yaml
import os

# =============================================================================
# CONFIGURATION - Update Paths for New System
# =============================================================================

# Base directory for the project
BASE_DIR = "/home/svarp/Desktop/Projects/ad2c - testEnv/AD2C-Diversity-Testing"

# Paths
ABS_CONFIG_PATH = f"{BASE_DIR}/het_control/conf"
CONFIG_NAME = "discovery_ippo_config"
SAVE_PATH = "/home/svarp/Desktop/Projects/ad2c - testEnv/model_checkpoint/discovery_ippo/"

# Training parameters
MAX_FRAMES = 3_000_000
CHECKPOINT_INTERVAL = 3_000_000

# Initial SND (matching your reference -1.0 value)
DESIRED_SND = 0.5

# Task-specific overrides
TASK_OVERRIDES = {
    # Add any specific discovery parameters here if needed
    "n_agents": 3,
    "n_targets": 5,
}

# ESC Controller Configuration
USE_ESC = True  
ESC_CONFIG_FILE = f"{BASE_DIR}/het_control/conf/callback/escontroller.yaml"

# Specific overrides for this task
ESC_OVERRIDES = {
    "control_group": "agents",
    "initial_snd": DESIRED_SND,
    "integrator_gain": -1.0,
}

# =============================================================================
# RUN EXPERIMENT
# =============================================================================

if __name__ == "__main__":
    # Load ESC config and apply overrides to create a temporary modified config
    temp_esc_config = None
    if USE_ESC and ESC_OVERRIDES:
        try:
            with open(ESC_CONFIG_FILE, 'r') as f:
                esc_config = yaml.safe_load(f)
            
            # Apply overrides to the 'esc_controller' block in the YAML
            if 'esc_controller' not in esc_config:
                esc_config['esc_controller'] = {}
                
            for key, value in ESC_OVERRIDES.items():
                esc_config['esc_controller'][key] = value
                print(f"🔧 Overriding ESC parameter: {key} = {value}")
            
            # Save to a task-specific temporary file
            temp_esc_config = "/tmp/escontroller_discovery.yaml"
            with open(temp_esc_config, 'w') as f:
                yaml.dump(esc_config, f)
            
            esc_config_to_use = temp_esc_config
        except FileNotFoundError:
            print(f"⚠️ Warning: ESC config not found at {ESC_CONFIG_FILE}. Using defaults.")
            esc_config_to_use = ESC_CONFIG_FILE
    else:
        esc_config_to_use = ESC_CONFIG_FILE
    
    print(f"{'='*80}")
    print(f"🎯 Running Discovery Task")
    print(f"{'='*80}\n")
    
    # Execute via the reusable run_experiment function from run.py
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