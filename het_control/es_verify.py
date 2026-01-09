"""
ESC Verification script that works without pandas.
"""
import wandb
import numpy as np
from collections import defaultdict


def verify_esc_run(run_path=None, run_id=None):
    """
    Verify ESC functionality from a WandB run (no pandas required).
    
    Args:
        run_path: WandB run path (e.g., "username/project/run_id") OR
        run_id: Just the run ID
    """
    api = wandb.Api()
    
    if run_id and not run_path:
        run_path = f"svarp-university-of-massachusetts-lowell/benchmarl/{run_id}"
    
    run = api.run(run_path)
    
    print("Fetching run history...")
    history_list = run.history(pandas=False)
    
    if not history_list:
        print("❌ ERROR: No history data found for this run!")
        return
    
    # Convert list of dicts to dict of lists
    data = defaultdict(list)
    for entry in history_list:
        for key, value in entry.items():
            if value is not None:  # Skip None values
                data[key].append(value)
    
    print("="*80)
    print(f"ESC VERIFICATION REPORT")
    print(f"Run: {run.name}")
    print(f"URL: {run.url}")
    print(f"History entries: {len(history_list)}")
    print("="*80)
    
    # Check 1: SND is changing
    print("\n1. Checking if SND is being updated...")
    if 'eval/agents/snd' in data:
        snd_values = np.array([v for v in data['eval/agents/snd'] if v is not None])
        if len(snd_values) > 0:
            snd_std = np.std(snd_values)
            snd_range = snd_values.max() - snd_values.min()
            
            print(f"   SND std: {snd_std:.4f}")
            print(f"   SND range: [{snd_values.min():.4f}, {snd_values.max():.4f}]")
            print(f"   SND range span: {snd_range:.4f}")
            
            if snd_std < 0.001:
                print("   ❌ FAIL: SND is not changing (std too low)")
            elif snd_values[0] == snd_values[-1]:
                print("   ⚠️  WARNING: SND starts and ends at same value")
            else:
                print("   ✅ PASS: SND is changing over time")
        else:
            print("   ❌ FAIL: No valid SND values found")
    else:
        print("   ❌ FAIL: 'eval/agents/snd' not found in logs")
    
    # Check 2: ESC controller is active
    print("\n2. Checking if ESC controller is active...")
    if 'esc/snd_actual' in data:
        esc_snd = np.array([v for v in data['esc/snd_actual'] if v is not None])
        if len(esc_snd) > 0:
            print(f"   ESC SND range: [{esc_snd.min():.4f}, {esc_snd.max():.4f}]")
            print("   ✅ PASS: ESC controller is logging")
        else:
            print("   ❌ FAIL: No valid ESC SND values")
    else:
        print("   ❌ FAIL: 'esc/snd_actual' not found - ESC may not be active")
    
    # Check 3: Gradient estimates are non-zero
    print("\n3. Checking gradient estimates...")
    if 'esc/gradient_estimate' in data:
        gradients = np.array([v for v in data['esc/gradient_estimate'] if v is not None])
        if len(gradients) > 0:
            grad_mean = np.mean(gradients)
            grad_std = np.std(gradients)
            non_zero_grads = np.sum(np.abs(gradients) > 1e-6)
            
            print(f"   Mean gradient: {grad_mean:.6f}")
            print(f"   Gradient std: {grad_std:.6f}")
            print(f"   Non-zero gradients: {non_zero_grads}/{len(gradients)}")
            
            if non_zero_grads == 0:
                print("   ❌ FAIL: All gradients are zero")
            elif non_zero_grads < len(gradients) * 0.5:
                print("   ⚠️  WARNING: More than 50% of gradients are zero")
            else:
                print("   ✅ PASS: Gradients are being computed")
        else:
            print("   ❌ FAIL: No valid gradient values")
    else:
        print("   ❌ FAIL: 'esc/gradient_estimate' not found")
    
    # Check 4: Perturbation is oscillating
    print("\n4. Checking perturbation signal...")
    if 'esc/perturbation_current' in data:
        perturbations = np.array([v for v in data['esc/perturbation_current'] if v is not None])
        if len(perturbations) > 0:
            pert_std = np.std(perturbations)
            pert_mean = np.abs(np.mean(perturbations))
            
            print(f"   Perturbation std: {pert_std:.4f}")
            print(f"   Perturbation mean (abs): {pert_mean:.4f}")
            
            if pert_std < 0.01:
                print("   ❌ FAIL: Perturbation not oscillating")
            elif pert_mean > pert_std:
                print("   ⚠️  WARNING: Perturbation may have DC bias")
            else:
                print("   ✅ PASS: Perturbation is oscillating")
        else:
            print("   ❌ FAIL: No valid perturbation values")
    else:
        print("   ❌ FAIL: 'esc/perturbation_current' not found")
    
    # Check 5: ESC Output vs Eval SND (CRITICAL CHECK)
    print("\n5. 🔍 CRITICAL: Checking ESC output vs actual evaluation SND...")
    if 'esc/snd_actual' in data and 'eval/agents/snd' in data:
        esc_snd = np.array([v for v in data['esc/snd_actual'] if v is not None])
        eval_snd = np.array([v for v in data['eval/agents/snd'] if v is not None])
        
        # Align arrays (take minimum length)
        min_len = min(len(esc_snd), len(eval_snd))
        
        if min_len > 0:
            esc_aligned = esc_snd[:min_len]
            eval_aligned = eval_snd[:min_len]
            
            difference = np.abs(esc_aligned - eval_aligned)
            correlation = np.corrcoef(esc_aligned, eval_aligned)[0, 1] if min_len > 1 else 0
            
            print(f"   ESC SND range: [{esc_aligned.min():.4f}, {esc_aligned.max():.4f}]")
            print(f"   Eval SND range: [{eval_aligned.min():.4f}, {eval_aligned.max():.4f}]")
            print(f"   Mean |ESC - Eval|: {difference.mean():.4f}")
            print(f"   Max |ESC - Eval|: {difference.max():.4f}")
            print(f"   Correlation: {correlation:.4f}")
            
            # Show ALL values side by side
            print(f"\n   📋 Complete ESC vs Eval Comparison (all {min_len} steps):")
            print(f"   {'Step':<6} {'ESC SND':>10} {'Eval SND':>10} {'Difference':>12} {'% Error':>10}")
            print(f"   {'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
            
            for i in range(min_len):
                diff = esc_aligned[i] - eval_aligned[i]
                pct_error = (abs(diff) / (esc_aligned[i] + 1e-8)) * 100 if esc_aligned[i] > 0.001 else 0
                marker = "  ⚠️" if abs(diff) > 0.2 else "  ✓" if abs(diff) < 0.05 else ""
                print(f"   {i:<6} {esc_aligned[i]:>10.4f} {eval_aligned[i]:>10.4f} {diff:>+12.4f} {pct_error:>9.1f}%{marker}")
            
            # Check for lag pattern
            print(f"\n   🔍 Checking for lag pattern...")
            if min_len > 2:
                # Compare eval[t] with esc[t-1]
                eval_vs_prev_esc = np.abs(eval_aligned[1:] - esc_aligned[:-1])
                direct_diff = difference[1:]
                
                print(f"   Direct comparison |ESC[t] - Eval[t]|: mean={direct_diff.mean():.4f}")
                print(f"   Lagged comparison |ESC[t-1] - Eval[t]|: mean={eval_vs_prev_esc.mean():.4f}")
                
                if eval_vs_prev_esc.mean() < direct_diff.mean() * 0.8:
                    print(f"   💡 DETECTED: Eval SND lags ESC by 1 step (this is normal in RL!)")
                    
                    # Recalculate correlation with lag
                    lagged_corr = np.corrcoef(esc_aligned[:-1], eval_aligned[1:])[0, 1] if min_len > 2 else 0
                    print(f"   Lagged correlation: {lagged_corr:.4f}")
                    
                    if lagged_corr > 0.9:
                        print(f"   ✅ EXCELLENT: With lag correction, tracking is very strong!")
            
            # Final verdict
            print(f"\n   📊 Final Assessment:")
            if difference.mean() > 0.5:
                print("   ❌ FAIL: Large mismatch between ESC output and eval SND")
                print("          The model may not be using the ESC-controlled diversity!")
            elif difference.mean() > 0.1:
                print("   ⚠️  WARNING: Moderate mismatch between ESC and eval SND")
                if min_len > 2 and eval_vs_prev_esc.mean() < difference.mean() * 0.8:
                    print("   ℹ️  However, this appears to be due to 1-step lag (normal behavior)")
            else:
                print("   ✅ PASS: ESC output matches evaluation SND")
                
            if correlation < 0.5:
                print("   ⚠️  WARNING: Low correlation - ESC and eval SND not tracking together")
            elif correlation > 0.9:
                print("   ✅ EXCELLENT: Strong correlation between ESC and eval SND")
            elif correlation > 0.7:
                print("   ✅ GOOD: Moderate-to-strong correlation between ESC and eval SND")
        else:
            print("   ❌ FAIL: No overlapping data points between ESC and eval")
    else:
        missing = []
        if 'esc/snd_actual' not in data:
            missing.append('esc/snd_actual')
        if 'eval/agents/snd' not in data:
            missing.append('eval/agents/snd')
        print(f"   ❌ FAIL: Missing metrics: {missing}")
    
    # Check 6: Setpoint vs Actual
    print("\n6. Checking setpoint vs actual SND...")
    if 'esc/snd_setpoint' in data and 'esc/snd_actual' in data:
        setpoint = np.array([v for v in data['esc/snd_setpoint'] if v is not None])
        actual = np.array([v for v in data['esc/snd_actual'] if v is not None])
        
        min_len = min(len(setpoint), len(actual))
        if min_len > 0:
            difference = np.abs(actual[:min_len] - setpoint[:min_len])
            print(f"   Mean |actual - setpoint|: {difference.mean():.4f}")
            print(f"   Max |actual - setpoint|: {difference.max():.4f}")
            print("   ✅ PASS: Both metrics are being tracked")
        else:
            print("   ⚠️  WARNING: No valid data in setpoint or actual")
    
    # Check 7: Reward correlation
    print("\n7. Checking reward trends...")
    if 'esc/reward_mean' in data:
        rewards = np.array([v for v in data['esc/reward_mean'] if v is not None])
        if len(rewards) > 0:
            print(f"   Initial reward: {rewards[0]:.4f}")
            print(f"   Final reward: {rewards[-1]:.4f}")
            print(f"   Reward change: {rewards[-1] - rewards[0]:+.4f}")
            
            if len(rewards) > 10:
                # Simple trend check
                x = np.arange(len(rewards))
                slope = np.polyfit(x, rewards, 1)[0]
                print(f"   Reward trend (slope): {slope:+.6f}")
                
                if slope > 0:
                    print("   ✅ Rewards are generally improving")
                elif slope < 0:
                    print("   ⚠️  Rewards are generally decreasing")
                else:
                    print("   ℹ️  Rewards are stable")
    
    # Check 8: List all available metrics
    print("\n8. Available metrics:")
    esc_metrics = [k for k in data.keys() if 'esc/' in k]
    eval_metrics = [k for k in data.keys() if 'eval/' in k]
    
    print(f"   ESC metrics ({len(esc_metrics)}):")
    for metric in sorted(esc_metrics)[:10]:  # Show first 10
        print(f"      - {metric}")
    if len(esc_metrics) > 10:
        print(f"      ... and {len(esc_metrics) - 10} more")
    
    print(f"   Eval metrics ({len(eval_metrics)}):")
    for metric in sorted(eval_metrics)[:10]:
        print(f"      - {metric}")
    if len(eval_metrics) > 10:
        print(f"      ... and {len(eval_metrics) - 10} more")
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
        print(f"Checking run: {run_id}\n")
        verify_esc_run(run_id=run_id)
    else:
        print("Usage: python es_verify.py <run_id>")
        print("\nExample:")
        print("  python es_verify.py ippo_discovery_hetcontrolmlpempirical__ae216f79_26_01_09-03_33_48")