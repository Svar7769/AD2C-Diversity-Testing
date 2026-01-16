"""
Runner script for Navigation task with ESC control.
Updated for easy ESC parameter configuration.
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
CONFIG_NAME = "navigation_ippo"
SAVE_PATH = "/home/spatel/Desktop/ad2c/model_checkpoint/navigation_ippo/"

# Training parameters
MAX_FRAMES = 12_000_000
CHECKPOINT_INTERVAL = 12_000_000

# Initial SND
DESIRED_SND = 0.0

# Task-specific overrides
TASK_OVERRIDES = {
    "n_agents": 3,
    "agents_with_same_goal": 3,
}

# =============================================================================
# ESC CONTROLLER CONFIGURATION
# =============================================================================
USE_ESC = True  # Set to False to disable ESC

# ESC Parameters - Easy to modify here!
ESC_PARAMS = {
    # Basic parameters
    "control_group": "agents",
    "initial_snd": DESIRED_SND,
    
    # Perturbation parameters
    "dither_magnitude": 0.2,      # Amplitude of perturbation (try 0.1-0.3)
    "dither_frequency": 1.0,      # Frequency in rad/s (ω) - fastest
    
    # Filter parameters (MUST maintain: ωh < ωl < ω)
    "high_pass_cutoff": 0.05,     # ωh (slowest) - must be < low_pass_cutoff
    "low_pass_cutoff": 0.5,       # ωl (medium) - must be > high_pass and < dither_frequency
    
    # Integration parameters
    "integrator_gain": -10,     # Negative for gradient descent (try -0.005 to -0.02)
    "sampling_period": 1.0,       # Time between ESC updates (seconds)
    
    # Output bounds
    "min_snd": 0.0,
    "max_snd": 3.0,
    
    # Adaptive features
    "use_adaptive_gain": True,         # Enable RMSprop-style adaptive gain
    "use_adaptive_dither": True,      # Enable dither annealing (exploration decay)
    
    # Adaptive gain parameters (only used if use_adaptive_gain=True)
    "gain_adaptation_mode": "rmsprop", # Options: "rmsprop", "binary", "gradient_norm"
    "binary_gain_threshold": 0.2,      # For binary mode
    "binary_high_gain_multiplier": 2.5, # For binary mode
    
    # Adaptive dither parameters (only used if use_adaptive_dither=True)
    "dither_decay_rate": 0.999,        # Decay rate (0.995-0.999, closer to 1 = slower)
    "min_dither_ratio": 0.1,           # Minimum dither (10% of initial)
    "dither_boost_threshold": 0.01,    # Boost dither if gradient < this
    "dither_boost_rate": 1.02,         # Boost rate (2% increase)
    
    # Action loss parameters
    "use_action_loss": False,
    "action_loss_lr": 0.001,
}

# =============================================================================
# QUICK PRESETS - Uncomment to use
# =============================================================================

# # Preset 1: Classical ESC (most conservative)
# ESC_PARAMS.update({
#     "use_adaptive_gain": False,
#     "use_adaptive_dither": False,
# })

# # Preset 2: Adaptive Gain Only (recommended)
# ESC_PARAMS.update({
#     "use_adaptive_gain": True,
#     "gain_adaptation_mode": "rmsprop",
#     "use_adaptive_dither": False,
# })

# # Preset 3: Full Adaptive (most aggressive)
# ESC_PARAMS.update({
#     "use_adaptive_gain": True,
#     "gain_adaptation_mode": "rmsprop",
#     "use_adaptive_dither": True,
#     "dither_decay_rate": 0.999,
# })

# # Preset 4: Fast Learning (higher gain & dither)
# ESC_PARAMS.update({
#     "integrator_gain": -0.02,        # Faster updates (less stable)
#     "dither_magnitude": 0.3,         # Larger exploration
#     "use_adaptive_gain": True,
# })

# # Preset 5: Conservative/Stable (lower gain & dither)
# ESC_PARAMS.update({
#     "integrator_gain": -0.005,       # Slower updates (more stable)
#     "dither_magnitude": 0.1,         # Smaller exploration
#     "use_adaptive_gain": True,
# })

# =============================================================================
# VALIDATION - Check frequency ordering
# =============================================================================
def validate_esc_params(params):
    """Validate ESC parameters before running."""
    h = params.get("high_pass_cutoff", 0.05)
    l = params.get("low_pass_cutoff", 0.5)
    d = params.get("dither_frequency", 1.0)
    
    if h >= l:
        raise ValueError(
            f"❌ ERROR: high_pass_cutoff ({h}) must be < low_pass_cutoff ({l})\n"
            f"   ESC theory requires: ωh < ωl < ω"
        )
    if l >= d:
        raise ValueError(
            f"❌ ERROR: low_pass_cutoff ({l}) must be < dither_frequency ({d})\n"
            f"   ESC theory requires: ωh < ωl < ω"
        )
    
    print(f"✅ Frequency ordering valid: ωh={h} < ωl={l} < ω={d}")
    return True

# =============================================================================
# RUN EXPERIMENT
# =============================================================================
if __name__ == "__main__":
    # Validate ESC parameters
    if USE_ESC:
        print("\n" + "="*80)
        print("🔧 ESC Configuration:")
        print("="*80)
        
        # Validate frequency ordering
        validate_esc_params(ESC_PARAMS)
        
        # Print key parameters
        print(f"\n📊 Key ESC Parameters:")
        print(f"   Initial SND: {ESC_PARAMS['initial_snd']}")
        print(f"   Dither: ±{ESC_PARAMS['dither_magnitude']} @ {ESC_PARAMS['dither_frequency']} rad/s")
        print(f"   Integrator gain: {ESC_PARAMS['integrator_gain']}")
        print(f"   Adaptive gain: {ESC_PARAMS['use_adaptive_gain']} ({ESC_PARAMS.get('gain_adaptation_mode', 'N/A')})")
        print(f"   Adaptive dither: {ESC_PARAMS['use_adaptive_dither']}")
        print("="*80 + "\n")
    
    # Create temporary ESC config with all parameters
    temp_esc_config = None
    if USE_ESC:
        try:
            # Create complete ESC config
            esc_config = {"esc_controller": ESC_PARAMS}
            
            # Save to temporary file
            temp_esc_config = "/tmp/escontroller_navigation.yaml"
            with open(temp_esc_config, 'w') as f:
                yaml.dump(esc_config, f)
            
            print(f"✅ ESC config saved to: {temp_esc_config}\n")
            esc_config_to_use = temp_esc_config
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to create ESC config: {e}")
            print("   Falling back to default config file.")
            esc_config_to_use = f"{BASE_DIR}/het_control/conf/callback/escontroller.yaml"
    else:
        esc_config_to_use = None
    
    print(f"{'='*80}")
    print(f"🎯 Running Navigation Task")
    print(f"{'='*80}\n")
    
    # Execute via the reusable run_experiment function from run.py
    run_experiment(
        config_path=ABS_CONFIG_PATH,
        config_name=CONFIG_NAME,
        save_path=SAVE_PATH,
        max_frames=MAX_FRAMES,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        # desired_snd=DESIRED_SND,
        task_overrides=TASK_OVERRIDES,
        esc_config_path=esc_config_to_use,
        use_esc=USE_ESC
    )