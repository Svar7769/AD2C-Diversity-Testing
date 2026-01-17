"""
Runner script for Simple Tag task with ESC control.
Example showing adversarial control group.
Path: /home/spatel/Desktop/ad2c/AD2C-Diversity-Testing/
"""
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
CONFIG_NAME = "tag_ippo_config"
SAVE_PATH = "/home/spatel/Desktop/ad2c/model_checkpoint/tag_ippo/"

# Training parameters
MAX_FRAMES = 12_000_000
CHECKPOINT_INTERVAL = 12_000_000

# Initial SND (will be overridden by ESC config)
DESIRED_SND = 0.0

# Task-specific overrides (keep only static parameters)
TASK_OVERRIDES = {
    "num_adversaries": 3,
}

# ESC Controller Configuration
USE_ESC = True  # Set to False to disable ESC
ESC_CONFIG_FILE = f"{BASE_DIR}/het_control/conf/callback/escontroller.yaml"

# Specific overrides for this task
# ⚠️ IMPORTANT: These values have been corrected for proper ESC operation with ESCCallback
ESC_OVERRIDES = {
    "control_group": "adversary",  # Controls the predators (adversaries)
    "initial_snd": 0.0,
    "dither_magnitude": 0.2,      
    "dither_frequency": 1.0,    

    # CORRECTED VALUES (proper frequency separation for ESCCallback):
    # Note: ESCCallback defaults are high_pass=0.1, low_pass=0.05
    "high_pass_cutoff": 0.1,      # ωh (high-pass cutoff)
    "low_pass_cutoff": 0.05,      # ωl (low-pass cutoff)
    
    "integrator_gain": -1.0,      # Base gain for gradient descent
    
    "sampling_period": 1.0,       
    "min_snd": 0.0,
    "max_snd": 3.0,
    "use_adaptive_gain": True,    # ESCCallback has built-in binary adaptive gain
    
    # Simple Tag specific parameters
    "simple_tag_freeze_policy": False,
    "simple_tag_freeze_policy_after_frames": 1_000_000,
    
    # Action loss parameters
    "use_action_loss": False,
    "action_loss_lr": 0.001,
}

# RUN EXPERIMENT
# =============================================================================
if __name__ == "__main__":
    # Load ESC config and apply overrides to create a temporary modified config
    temp_esc_config = None
    if USE_ESC and ESC_OVERRIDES:
        try:
            with open(ESC_CONFIG_FILE, 'r') as f:
                esc_config = yaml.safe_load(f)
            
            if 'esc_controller' not in esc_config:
                esc_config['esc_controller'] = {}
            
            for key, value in ESC_OVERRIDES.items():
                esc_config['esc_controller'][key] = value
                print(f"🔧 Overriding ESC parameter: {key} = {value}")
            
            # Save to a task-specific temporary file
            temp_esc_config = "/tmp/escontroller_simple_tag.yaml"
            with open(temp_esc_config, 'w') as f:
                yaml.dump(esc_config, f)
            
            esc_config_to_use = temp_esc_config
            
        except FileNotFoundError:
            print(f"⚠️  Warning: ESC config not found at {ESC_CONFIG_FILE}. Using defaults.")
            esc_config_to_use = ESC_CONFIG_FILE
    else:
        esc_config_to_use = ESC_CONFIG_FILE
    
    print(f"{'='*80}")
    print(f"🎯 Running Simple Tag Task (Adversarial ESC)")
    print(f"{'='*80}")
    print(f"📊 ESC Configuration:")
    print(f"   Control group: {ESC_OVERRIDES['control_group']}")
    print(f"   Frequency ordering: ωl={ESC_OVERRIDES['low_pass_cutoff']} < ωh={ESC_OVERRIDES['high_pass_cutoff']} < ω={ESC_OVERRIDES['dither_frequency']}")
    print(f"   Base gain: {ESC_OVERRIDES['integrator_gain']}")
    if ESC_OVERRIDES['use_adaptive_gain']:
        print(f"   Adaptive gain: Binary switching (built-in)")
    print(f"   Policy freeze: {'Yes' if ESC_OVERRIDES['simple_tag_freeze_policy'] else 'No'} (after {ESC_OVERRIDES['simple_tag_freeze_policy_after_frames']:,} frames)")
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