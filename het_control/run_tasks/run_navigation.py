"""
Runner script for Navigation task with ESC control.
Fully dynamic: Supports any CLI overrides for task, model, or experiment.
"""
from het_control.run import run_experiment
import yaml
import os
import sys

# =============================================================================
# CONFIGURATION - Paths
# =============================================================================
BASE_DIR = "/home/svarp/Desktop/Projects/ad2c - testEnv/AD2C-Diversity-Testing/"
ABS_CONFIG_PATH = f"{BASE_DIR}/het_control/conf"
CONFIG_NAME = "navigation_ippo"
BASE_SAVE_PATH = "/home/svarp/Desktop/Projects/ad2c - testEnv/model_checkpoint/navigation_ippo"

# Default Training parameters (can be overridden via CLI)
MAX_FRAMES = 12_000_000
CHECKPOINT_INTERVAL = 12_000_000
USE_ESC = True

# Task-specific defaults
TASK_OVERRIDES = {"n_agents": 3}

# ESC Defaults
ESC_PARAMS = {
    "control_group": "agents",
    "initial_snd": 0.0,
    "dither_magnitude": 0.2,      
    "dither_frequency": 1.0,      
    "high_pass_cutoff": 0.05,     
    "low_pass_cutoff": 0.5,       
    "integrator_gain": -10,     
    "sampling_period": 1.0,       
    "min_snd": 0.0,
    "max_snd": 3.0,
    "use_adaptive_gain": True,         
    "gain_adaptation_mode": "rmsprop", 
    "use_adaptive_dither": True,
}

# =============================================================================
# DYNAMIC OVERRIDE LOGIC
# =============================================================================
if __name__ == "__main__":
    # This loop catches ANY argument passed like key=value
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=")
            
            # Type Conversion (int, float, bool, or string)
            if value.lower() == "true": value = True
            elif value.lower() == "false": value = False
            elif value.replace(".", "", 1).isdigit():
                value = float(value) if "." in value else int(value)

            # 1. Handle Task Overrides (task.anything=value)
            if key.startswith("task."):
                task_key = key.split(".")[1]
                TASK_OVERRIDES[task_key] = value
                print(f"🔧 Task Override: {task_key} = {value}")

            # 2. Handle ESC/SND Overrides (model.desired_snd=value)
            elif key == "model.desired_snd":
                ESC_PARAMS["initial_snd"] = value
                print(f"🔧 ESC Start SND: {value}")

            # 3. Handle Script Overrides (e.g., max_frames=50000)
            elif key == "max_frames":
                MAX_FRAMES = value
            elif key == "checkpoint_interval":
                CHECKPOINT_INTERVAL = value

    # Create dynamic Save Path name based on CLI inputs
    # This helps distinguish runs in your folders
    current_goals = TASK_OVERRIDES.get("agents_with_same_goal", "std")
    current_snd = ESC_PARAMS["initial_snd"]
    specific_save_path = os.path.join(BASE_SAVE_PATH, f"snd_{current_snd}_goals_{current_goals}")

    os.makedirs(specific_save_path, exist_ok=True)
    
    # --- ESC SETUP ---
    if USE_ESC:
        temp_esc_config = "/tmp/escontroller_navigation.yaml"
        with open(temp_esc_config, 'w') as f:
            yaml.dump({"esc_controller": ESC_PARAMS}, f)
        esc_config_to_use = temp_esc_config
    else:
        esc_config_to_use = None

    print(f"\n🚀 Launching Experiment | SND: {current_snd} | Goals: {current_goals}")
    print(f"📂 Saving to: {specific_save_path}\n")

    # Execute
    run_experiment(
        config_path=ABS_CONFIG_PATH,
        config_name=CONFIG_NAME,
        save_path=specific_save_path,
        max_frames=MAX_FRAMES,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        task_overrides=TASK_OVERRIDES,
        esc_config_path=esc_config_to_use,
        use_esc=USE_ESC
    )