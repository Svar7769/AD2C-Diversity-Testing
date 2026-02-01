"""
Test your actual ESC implementation with real filter dynamics.
Includes parameter sweep functionality to visualize how different
cutoff frequencies affect filter performance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from het_control.callbacks.esc_controller import ExtremumSeekingController


def replay_with_real_esc(rewards, config):
    """
    Replay logged reward data through your actual ESC implementation.

    Args:
        rewards: List of reward values from training
        config: Dictionary with ESC configuration

    Returns:
        Dictionary with all ESC internal states over time
    """
    # Initialize ESC with given config
    esc = ExtremumSeekingController(
        sampling_period=config['sampling_period'],
        dither_frequency=config['dither_frequency'],
        dither_magnitude=config['dither_magnitude'],
        integrator_gain=config['integrator_gain'],
        initial_value=config['initial_snd'],
        high_pass_cutoff=config['high_pass_cutoff'],
        low_pass_cutoff=config['low_pass_cutoff'],
        use_adaptive_gain=config['use_adaptive_gain'],
        min_output=config['min_snd'],
        max_output=config['max_snd']
    )

    # Storage for ESC states
    history = {
        'costs': [],
        'rewards': [],
        'output': [],           # Perturbed SND
        'setpoint': [],         # Setpoint without perturbation
        'perturbation': [],
        'hpf_output': [],
        'lpf_output': [],       # Gradient estimate
        'gradient_magnitude': [],
        'phase': [],
        'integral': [],
        'using_high_gain': [],
    }

    # Run ESC update for each reward
    for reward in rewards:
        cost = -reward  # ESC minimizes cost

        # Update ESC
        (
            output,
            hpf_output,
            lpf_output,
            gradient_magnitude,
            _,
            setpoint
        ) = esc.update(cost)

        # Store states
        history['costs'].append(cost)
        history['rewards'].append(reward)
        history['output'].append(output)
        history['setpoint'].append(setpoint)
        history['perturbation'].append(output - setpoint)
        history['hpf_output'].append(hpf_output)
        history['lpf_output'].append(lpf_output)
        history['gradient_magnitude'].append(gradient_magnitude)
        history['phase'].append(esc.phase)
        history['integral'].append(esc.integral)

        # Check if high gain is active
        using_high_gain = (
            esc.use_adaptive and
            gradient_magnitude > esc.gradient_threshold
        )
        history['using_high_gain'].append(using_high_gain)

    # Convert to numpy arrays
    for key in history:
        history[key] = np.array(history[key])

    return history, esc


def analyze_esc_performance(history, title="ESC Performance"):
    """
    Analyze and print metrics from ESC history.
    """
    n_steps = len(history['rewards'])

    print("="*70)
    print(f"{title}")
    print("="*70)

    # Check 1: DC Removal
    cost_mean = np.mean(history['costs'])
    cost_std = np.std(history['costs'])
    hpf_mean = np.mean(history['hpf_output'])

    print(f"\n1. HIGH-PASS FILTER (DC Removal)")
    print(f"   Cost Mean:     {cost_mean:.4f}")
    print(f"   Cost Std:      {cost_std:.4f}")
    print(f"   HPF Mean:      {hpf_mean:.4f} (should be ~0)")
    print(f"   DC Removed:    {abs(hpf_mean) < 0.1 * cost_std}")

    # Check 2: Gradient correlation
    if n_steps > 3:
        # Compute finite difference gradients at perturbation zero-crossings
        true_grads = []
        est_grads = []

        perturbations = history['perturbation']
        for i in range(1, n_steps - 1):
            if perturbations[i-1] * perturbations[i+1] < 0:
                true_grad = (history['costs'][i+1] - history['costs'][i-1]) / \
                           (perturbations[i+1] - perturbations[i-1])
                true_grads.append(true_grad)
                est_grads.append(history['lpf_output'][i])

        if len(true_grads) > 1:
            true_grads = np.array(true_grads)
            est_grads = np.array(est_grads)
            correlation = np.corrcoef(true_grads, est_grads)[0, 1]
            true_rms = np.sqrt(np.mean(true_grads**2))
            est_rms = np.sqrt(np.mean(est_grads**2))

            print(f"\n2. GRADIENT QUALITY")
            print(f"   Correlation:     {correlation:.4f} (>0.5 is good)")
            print(f"   True RMS:        {true_rms:.4f}")
            print(f"   Estimated RMS:   {est_rms:.4f}")
            print(f"   RMS Ratio:       {est_rms/true_rms:.4f} (should be ~1.0)")
            print(f"   Attenuation:     {1/(est_rms/true_rms):.2f}x")

    # Check 3: Convergence
    initial_reward = history['rewards'][0]
    final_reward = np.mean(history['rewards'][-5:])  # Last 5 average
    improvement = final_reward - initial_reward

    print(f"\n3. CONVERGENCE")
    print(f"   Initial Reward:  {initial_reward:.4f}")
    print(f"   Final Reward:    {final_reward:.4f}")
    print(f"   Improvement:     {improvement:.4f}")
    print(f"   Final Setpoint:  {history['setpoint'][-1]:.4f}")

    # Check 4: Adaptive gain usage
    high_gain_count = np.sum(history['using_high_gain'])
    print(f"\n4. ADAPTIVE GAIN")
    print(f"   High Gain Used:  {high_gain_count}/{n_steps} steps")
    print(f"   Percentage:      {100*high_gain_count/n_steps:.1f}%")

    print("="*70 + "\n")

    return {
        'dc_removed': abs(hpf_mean) < 0.1 * cost_std,
        'correlation': correlation if 'correlation' in locals() else 0.0,
        'attenuation': (1/(est_rms/true_rms)) if 'est_rms' in locals() and est_rms > 0 else 0.0,
        'improvement': improvement,
    }


def plot_esc_internals(history, config, filename='real_esc_analysis.png'):
    """
    Create comprehensive plot of ESC internal states.
    """
    n_steps = len(history['rewards'])
    steps = np.arange(n_steps)

    fig, axes = plt.subplots(5, 1, figsize=(14, 12))

    config_str = (f"Dither: {config['dither_frequency']:.2f} rad/s, "
                  f"HPF: {config['high_pass_cutoff']:.2f} rad/s, "
                  f"LPF: {config['low_pass_cutoff']:.2f} rad/s, "
                  f"Gain: {config['integrator_gain']:.2f}")
    fig.suptitle(f'ESC Internal States\n{config_str}',
                 fontsize=14, fontweight='bold')

    # Plot 1: Rewards and Costs
    ax1 = axes[0]
    ax1.plot(steps, history['rewards'], 'b-', label='Reward', linewidth=2)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(steps, history['costs'], 'r--', label='Cost', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('Reward', color='b')
    ax1_twin.set_ylabel('Cost', color='r')
    ax1.set_title('Reward/Cost Over Time')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: SND Values
    ax2 = axes[1]
    ax2.plot(steps, history['output'], 'purple', label='Output (with perturbation)',
             linewidth=2, alpha=0.7)
    ax2.plot(steps, history['setpoint'], 'b-', label='Setpoint', linewidth=2)
    ax2.fill_between(steps,
                      history['setpoint'] - config['dither_magnitude'],
                      history['setpoint'] + config['dither_magnitude'],
                      alpha=0.2, color='gray', label='Perturbation range')
    ax2.set_ylabel('SND Value')
    ax2.set_title('SND Evolution (Output vs Setpoint)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Filter Outputs
    ax3 = axes[2]
    ax3.plot(steps, history['hpf_output'], 'g-', label='HPF Output', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_ylabel('HPF Output')
    ax3.set_title('High-Pass Filter Output (DC Removed)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Gradient Estimation
    ax4 = axes[3]
    # Compute demodulated signal
    demodulated = history['hpf_output'] * np.sin(history['phase'])
    ax4.plot(steps, demodulated, 'orange', label='Demodulated (HPF × sin)',
             alpha=0.5, linewidth=1)
    ax4.plot(steps, history['lpf_output'], 'purple', label='Gradient (LPF output)',
             linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Highlight high-gain regions
    high_gain_steps = steps[history['using_high_gain']]
    if len(high_gain_steps) > 0:
        ax4.scatter(high_gain_steps,
                   history['lpf_output'][history['using_high_gain']],
                   c='red', s=50, marker='x', label='High Gain Active', zorder=5)

    ax4.set_ylabel('Signal')
    ax4.set_title('Gradient Estimation (Demodulation + LPF)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Gradient Magnitude and Setpoint Changes
    ax5 = axes[4]
    ax5.plot(steps, history['gradient_magnitude'], 'r-',
             label='Gradient RMS', linewidth=2)
    ax5.axhline(y=0.2, color='orange', linestyle='--',
                label='High-gain threshold', alpha=0.5)

    # Show setpoint changes
    setpoint_changes = np.concatenate([[0], np.diff(history['setpoint'])])
    ax5_twin = ax5.twinx()
    ax5_twin.plot(steps, setpoint_changes, 'b--',
                  label='Setpoint Change', linewidth=1.5, alpha=0.7)

    ax5.set_ylabel('Gradient Magnitude', color='r')
    ax5_twin.set_ylabel('Setpoint Change', color='b')
    ax5.set_xlabel('Evaluation Step')
    ax5.set_title('Gradient Magnitude and Setpoint Adaptation')
    ax5.legend(loc='upper left')
    ax5_twin.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(__file__).parent / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def parameter_sweep(rewards, base_config, param_name, param_values):
    """
    Test how different parameter values affect ESC performance.

    Args:
        rewards: List of reward values
        base_config: Base ESC configuration
        param_name: Name of parameter to sweep (e.g., 'high_pass_cutoff')
        param_values: List of values to test for that parameter

    Returns:
        Dictionary mapping param values to results
    """
    results = {}

    print("\n" + "="*70)
    print(f"PARAMETER SWEEP: {param_name}")
    print("="*70 + "\n")

    for value in param_values:
        # Create config with modified parameter
        config = base_config.copy()
        config[param_name] = value

        print(f"\nTesting {param_name} = {value}:")
        print("-" * 40)

        # Run ESC with this configuration
        history, _ = replay_with_real_esc(rewards, config)

        # Analyze performance
        metrics = analyze_esc_performance(history, title=f"{param_name}={value}")

        results[value] = {
            'history': history,
            'metrics': metrics,
            'config': config
        }

    return results


def plot_parameter_comparison(sweep_results, param_name,
                              filename='parameter_comparison.png'):
    """
    Create comparison plots for parameter sweep results.
    """
    param_values = sorted(sweep_results.keys())
    n_configs = len(param_values)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Parameter Sweep: {param_name}', fontsize=14, fontweight='bold')

    # Extract metrics
    attenuations = [sweep_results[v]['metrics']['attenuation'] for v in param_values]
    correlations = [sweep_results[v]['metrics']['correlation'] for v in param_values]
    improvements = [sweep_results[v]['metrics']['improvement'] for v in param_values]
    final_setpoints = [sweep_results[v]['history']['setpoint'][-1] for v in param_values]

    # Plot 1: Gradient Attenuation
    ax1 = axes[0, 0]
    ax1.plot(param_values, attenuations, 'o-', linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color='green', linestyle='--', label='Ideal (1.0x)', alpha=0.5)
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Gradient Attenuation (x)')
    ax1.set_title('Filter Attenuation vs Parameter')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gradient Correlation
    ax2 = axes[0, 1]
    ax2.plot(param_values, correlations, 's-', linewidth=2, markersize=8, color='purple')
    ax2.axhline(y=0.5, color='orange', linestyle='--', label='Good threshold', alpha=0.5)
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Gradient Correlation')
    ax2.set_title('Gradient Quality vs Parameter')
    ax2.set_ylim([-1, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Reward Improvement
    ax3 = axes[1, 0]
    ax3.bar(range(n_configs), improvements, color='teal', alpha=0.7)
    ax3.set_xticks(range(n_configs))
    ax3.set_xticklabels([f'{v:.2f}' for v in param_values], rotation=45)
    ax3.set_xlabel(param_name)
    ax3.set_ylabel('Reward Improvement')
    ax3.set_title('Optimization Performance vs Parameter')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Setpoint Evolution for All Configs
    ax4 = axes[1, 1]
    for i, value in enumerate(param_values):
        history = sweep_results[value]['history']
        steps = np.arange(len(history['setpoint']))
        label = f'{param_name}={value:.2f}'
        ax4.plot(steps, history['setpoint'], label=label, linewidth=2, alpha=0.7)

    ax4.set_xlabel('Evaluation Step')
    ax4.set_ylabel('Setpoint')
    ax4.set_title('Setpoint Convergence Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(__file__).parent / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_path}")
    plt.close()


def main():
    """
    Main analysis: Test your actual ESC with your logged data.
    """
    # Your logged reward data
    log_data = [
        # Step, Reward, SND_actual, Setpoint, Gradient, RMS
        [0,  0.632, 0.0000, 0.0000, 0.00000, 0.00000],
        [1,  1.723, 0.4643, 0.2960, -0.29604, 0.13239],
        [2,  2.108, 0.8733, 0.6915, -0.39543, 0.21283],
        [3,  2.785, 0.9605, 0.9323, -0.24080, 0.21871],
        [4,  2.905, 0.7631, 0.9145, 0.01780, 0.19578],
        [5,  2.919, 0.6229, 0.8147, 0.09979, 0.18071],
        [6,  3.120, 0.6816, 0.7375, 0.07723, 0.16528],
        [7,  2.951, 0.8345, 0.7031, 0.03434, 0.14863],
        [8,  2.990, 0.8967, 0.6988, 0.00431, 0.13295],
        [9,  3.026, 0.7855, 0.7031, -0.00425, 0.11893],
        [10, 3.170, 0.5725, 0.6814, 0.02171, 0.10681],
        [11, 3.058, 0.4767, 0.6767, 0.00465, 0.09556],
        [12, 3.047, 0.5701, 0.6774, -0.00072, 0.08547],
        [13, 3.207, 0.7774, 0.6933, -0.01591, 0.07678],
        [14, 3.095, 0.8901, 0.6920, 0.00136, 0.06868],
        [15, 2.974, 0.7987, 0.6686, 0.02336, 0.06231],
        [16, 2.965, 0.6050, 0.6626, 0.00600, 0.05579],
        [17, 3.096, 0.4455, 0.6377, 0.02487, 0.05113],
        [18, 3.091, 0.4674, 0.6176, 0.02010, 0.04661],
        [19, 3.152, 0.6406, 0.6106, 0.00699, 0.04180],
        [20, 3.151, 0.7987, 0.6161, -0.00545, 0.03747],
        [21, 3.125, 0.7850, 0.6176, -0.00154, 0.03352],
        [22, 3.107, 0.6167, 0.6185, -0.00082, 0.02998],
        [23, 3.153, 0.4420, 0.6112, 0.00722, 0.02701],
        [24, 3.118, 0.4304, 0.6115, -0.00026, 0.02416],
        [25, 3.146, 0.5913, 0.6053, 0.00311, 0.02163],
    ]

    data = np.array(log_data)
    rewards = data[:, 1]

    # Your current ESC configuration
    base_config = {
        'sampling_period': 1.0,
        'dither_frequency': 1.0,
        'dither_magnitude': 0.2,
        'integrator_gain': -1.0,
        'initial_snd': 0.0,
        'high_pass_cutoff': 1.0,
        'low_pass_cutoff': 1.0,
        'use_adaptive_gain': True,
        'min_snd': 0.0,
        'max_snd': 3.0,
    }

    print("\n" + "="*70)
    print("TESTING YOUR ACTUAL ESC IMPLEMENTATION")
    print("="*70)

    # Test 1: Current configuration
    print("\n1. CURRENT CONFIGURATION")
    history_current, _ = replay_with_real_esc(rewards, base_config)
    metrics_current = analyze_esc_performance(history_current, "Current Config")
    plot_esc_internals(history_current, base_config, 'real_esc_current.png')

    # Test 2: Parameter sweep - filter cutoffs
    print("\n2. PARAMETER SWEEP: Filter Cutoffs")
    cutoff_values = [0.5, 0.7, 1.0, 2.0]

    # Sweep high-pass cutoff
    print("\n--- Sweeping HIGH-PASS Cutoff ---")
    hpf_results = parameter_sweep(rewards, base_config, 'high_pass_cutoff', cutoff_values)
    plot_parameter_comparison(hpf_results, 'high_pass_cutoff', 'hpf_cutoff_comparison.png')

    # Sweep low-pass cutoff
    print("\n--- Sweeping LOW-PASS Cutoff ---")
    lpf_results = parameter_sweep(rewards, base_config, 'low_pass_cutoff', cutoff_values)
    plot_parameter_comparison(lpf_results, 'low_pass_cutoff', 'lpf_cutoff_comparison.png')

    # Test 3: Recommended configuration
    print("\n3. RECOMMENDED CONFIGURATION")
    recommended_config = base_config.copy()
    recommended_config['high_pass_cutoff'] = 0.5  # 5x separation
    recommended_config['low_pass_cutoff'] = 0.2   # 5x separation
    recommended_config['integrator_gain'] = -1.0  # Compensate for less attenuation

    history_recommended, _ = replay_with_real_esc(rewards, recommended_config)
    metrics_recommended = analyze_esc_performance(history_recommended, "Recommended Config")
    plot_esc_internals(history_recommended, recommended_config, 'real_esc_recommended.png')

    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"\nCurrent Config:")
    print(f"  Attenuation:    {metrics_current['attenuation']:.2f}x")
    print(f"  Correlation:    {metrics_current['correlation']:.3f}")
    print(f"  Improvement:    {metrics_current['improvement']:.3f}")

    print(f"\nRecommended Config:")
    print(f"  Attenuation:    {metrics_recommended['attenuation']:.2f}x")
    print(f"  Correlation:    {metrics_recommended['correlation']:.3f}")
    print(f"  Improvement:    {metrics_recommended['improvement']:.3f}")

    print("\n" + "="*70)
    print("Analysis complete! Check the generated PNG files.")
    print("="*70)


if __name__ == "__main__":
    main()
