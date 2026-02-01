"""
Diagnostic tool to check if ESC filters are properly extracting gradients.
Analyzes the relationship between perturbation, cost, and gradient estimates.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_esc_filters(
    costs,              # List of cost values (-reward)
    perturbations,      # List of perturbation values at each step
    gradients,          # List of gradient estimates from LPF
    hpf_outputs,        # List of HPF outputs
    phases,             # List of phase values
    setpoints,          # List of setpoint values
):
    """
    Analyze ESC filter performance.

    Returns diagnostic metrics and plots.
    """
    costs = np.array(costs)
    perturbations = np.array(perturbations)
    gradients = np.array(gradients)
    hpf_outputs = np.array(hpf_outputs)
    phases = np.array(phases)
    setpoints = np.array(setpoints)

    n_steps = len(costs)
    steps = np.arange(n_steps)

    # =================================================================
    # Check 1: HPF is removing DC bias
    # =================================================================
    cost_mean = np.mean(costs)
    cost_std = np.std(costs)
    hpf_mean = np.mean(hpf_outputs)
    hpf_std = np.std(hpf_outputs)

    print("="*70)
    print("CHECK 1: High-Pass Filter - DC Removal")
    print("="*70)
    print(f"Raw Cost:        Mean={cost_mean:.4f}, Std={cost_std:.4f}")
    print(f"After HPF:       Mean={hpf_mean:.4f}, Std={hpf_std:.4f}")
    print(f"DC Removed:      {abs(hpf_mean) < 0.1 * cost_std}")
    print(f"  → HPF mean should be near 0 if DC is removed")
    print()

    # =================================================================
    # Check 2: Correlation between perturbation and cost changes
    # =================================================================
    if n_steps > 2:
        cost_changes = np.diff(costs)
        pert_changes = np.diff(perturbations)  # Same length as cost_changes

        if len(cost_changes) > 1 and len(pert_changes) > 1:
            correlation = np.corrcoef(pert_changes, cost_changes)[0, 1]
        else:
            correlation = 0.0
    else:
        correlation = 0.0

    print("="*70)
    print("CHECK 2: Perturbation-Cost Correlation")
    print("="*70)
    print(f"Correlation: {correlation:.4f}")
    print(f"Expected: Non-zero if perturbations affect cost")
    print(f"  → Positive correlation: increasing SND increases cost (bad)")
    print(f"  → Negative correlation: increasing SND decreases cost (good)")
    print()

    # =================================================================
    # Check 3: Gradient estimates align with actual reward changes
    # =================================================================
    print("="*70)
    print("CHECK 3: Gradient Estimate Quality")
    print("="*70)

    # Compute "true" gradient by finite difference (when perturbation direction changes)
    # Look for points where perturbation switches sign
    true_gradients = []
    estimated_gradients = []

    for i in range(1, n_steps - 1):
        # Check if we crossed from positive to negative perturbation (or vice versa)
        if perturbations[i-1] * perturbations[i+1] < 0:
            # Approximate gradient as cost difference over perturbation difference
            true_grad = (costs[i+1] - costs[i-1]) / (perturbations[i+1] - perturbations[i-1])
            true_gradients.append(true_grad)
            estimated_gradients.append(gradients[i])

    if len(true_gradients) > 0:
        true_gradients = np.array(true_gradients)
        estimated_gradients = np.array(estimated_gradients)

        # Check correlation
        if len(true_gradients) > 1:
            grad_correlation = np.corrcoef(true_gradients, estimated_gradients)[0, 1]
        else:
            grad_correlation = 0.0

        # Check magnitude alignment
        true_rms = np.sqrt(np.mean(true_gradients**2))
        estimated_rms = np.sqrt(np.mean(estimated_gradients**2))

        print(f"Gradient Correlation: {grad_correlation:.4f}")
        print(f"True Gradient RMS:      {true_rms:.4f}")
        print(f"Estimated Gradient RMS: {estimated_rms:.4f}")
        print(f"RMS Ratio: {estimated_rms/true_rms:.4f}")
        print(f"  → Correlation > 0.5 suggests good gradient estimation")
        print(f"  → RMS ratio should be O(1) for proper scaling")
    else:
        print("Not enough data to compute finite-difference gradients")
    print()

    # =================================================================
    # Check 4: Demodulation effectiveness
    # =================================================================
    print("="*70)
    print("CHECK 4: Demodulation (HPF × sin(phase))")
    print("="*70)

    # Manually compute demodulation
    demodulated = hpf_outputs * np.sin(phases)
    demod_mean = np.mean(demodulated)
    demod_std = np.std(demodulated)

    print(f"Demodulated Signal: Mean={demod_mean:.4f}, Std={demod_std:.4f}")
    print(f"Gradient Estimate:  Mean={np.mean(gradients):.4f}, Std={np.std(gradients):.4f}")
    print(f"  → LPF should smooth demodulated signal")
    print()

    # =================================================================
    # Check 5: Filter frequency response
    # =================================================================
    print("="*70)
    print("CHECK 5: Filter Cutoff vs Dither Frequency")
    print("="*70)
    print("With cutoff_freq = dither_freq:")
    print("  → Filters at critical frequency (-3dB attenuation)")
    print("  → May allow some dither leakage into gradient estimate")
    print("  → Recommendation: cutoff_freq = dither_freq / 5")
    print()

    # =================================================================
    # Generate Diagnostic Plots
    # =================================================================
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle('ESC Filter Diagnostics', fontsize=14, fontweight='bold')

    # Plot 1: Cost and Perturbation
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.plot(steps, costs, 'b-', label='Cost (-Reward)', linewidth=2)
    ax1_twin.plot(steps, perturbations, 'r--', label='Perturbation', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('Cost', color='b')
    ax1_twin.set_ylabel('Perturbation', color='r')
    ax1.set_title('Raw Cost vs Perturbation')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: HPF Output
    ax2 = axes[1]
    ax2.plot(steps, hpf_outputs, 'g-', label='HPF Output', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_ylabel('HPF Output')
    ax2.set_title('High-Pass Filtered Cost (DC Removed)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Demodulation
    ax3 = axes[2]
    ax3.plot(steps, demodulated, 'orange', label='HPF × sin(phase)', alpha=0.6, linewidth=1)
    ax3.plot(steps, gradients, 'purple', label='LPF Output (Gradient)', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_ylabel('Signal')
    ax3.set_title('Demodulation and Low-Pass Filtering')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Setpoint and Gradient
    ax4 = axes[3]
    ax4_twin = ax4.twinx()
    ax4.plot(steps, setpoints, 'b-', label='Setpoint', linewidth=2)
    ax4_twin.plot(steps, gradients, 'r-', label='Gradient Estimate', linewidth=1.5, alpha=0.7)
    ax4.set_ylabel('Setpoint', color='b')
    ax4_twin.set_ylabel('Gradient', color='r')
    ax4.set_xlabel('Evaluation Step')
    ax4.set_title('Setpoint Evolution and Gradient Feedback')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = Path(__file__).parent / 'esc_filter_diagnostic.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Diagnostic plot saved to: {output_path}")
    plt.close()

    return {
        'cost_mean': cost_mean,
        'hpf_mean': hpf_mean,
        'correlation': correlation,
        'dc_removed': abs(hpf_mean) < 0.1 * cost_std,
    }


def example_from_logs():
    """
    Example: Parse data from your training logs.
    You can extract these from wandb or your console output.
    """
    # Example data from your logs (Steps 0-25)
    # Format: [step, reward, snd_actual, setpoint, gradient, gradient_rms]
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
    costs = -rewards  # ESC minimizes cost = -reward
    snd_actual = data[:, 2]
    setpoints = data[:, 3]
    gradients = data[:, 4]

    # Compute perturbations (actual - setpoint)
    perturbations = snd_actual - setpoints

    # Need to reconstruct HPF output
    # We don't have it directly, but we can estimate from the ESC loop
    # For now, use a simplified approximation
    hpf_outputs = costs - np.mean(costs)  # Simplified: just remove DC

    # Reconstruct phases (assuming dither_frequency = 1.0 rad/s, sampling_period = 1.0)
    omega = 1.0
    dt = 1.0
    phases = np.array([i * omega * dt for i in range(len(data))]) % (2 * np.pi)

    print("\n" + "="*70)
    print("ANALYZING YOUR ESC DATA FROM LOGS")
    print("="*70 + "\n")

    results = analyze_esc_filters(
        costs=costs,
        perturbations=perturbations,
        gradients=gradients,
        hpf_outputs=hpf_outputs,
        phases=phases,
        setpoints=setpoints,
    )

    return results


if __name__ == "__main__":
    # Run analysis on your log data
    results = example_from_logs()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"DC Removed: {results['dc_removed']}")
    print(f"HPF Mean: {results['hpf_mean']:.4f} (should be near 0)")
    print(f"Perturbation-Cost Correlation: {results['correlation']:.4f}")
    print("\nSee esc_filter_diagnostic.png for visual analysis")
