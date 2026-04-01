"""
Smart Adaptive Extremum Seeking Control (ESC) implementation.
Automatically adapts perturbation magnitude and integrator gain based on convergence state.
"""
import numpy as np
from typing import Tuple, Optional
from enum import Enum


class ConvergenceState(Enum):
    """Controller operational states."""
    EXPLORATION = "exploration"  # Initial search, large perturbations
    SEEKING = "seeking"          # Approaching optimum, moderate perturbations
    CONVERGED = "converged"      # Near optimum, small perturbations
    TRACKING = "tracking"        # Following slow changes, minimal perturbations


class HighPassFilter:
    """First-order high-pass filter to remove DC offset from signals."""
    
    def __init__(self, sampling_period: float, cutoff_frequency: float):
        self.dt = sampling_period
        self.wc = cutoff_frequency
        self.alpha = self.wc / (self.wc + 1.0 / self.dt)
        self.prev_input = 0.0
        self.prev_output = 0.0
    
    def apply(self, input_signal: float) -> float:
        output = self.alpha * (self.prev_output + input_signal - self.prev_input)
        self.prev_input = input_signal
        self.prev_output = output
        return output
    
    def reset(self):
        self.prev_input = 0.0
        self.prev_output = 0.0


class LowPassFilter:
    """First-order low-pass filter to smooth signals."""
    
    def __init__(self, sampling_period: float, cutoff_frequency: float):
        self.dt = sampling_period
        self.wc = cutoff_frequency
        self.alpha = (self.dt * self.wc) / (1.0 + self.dt * self.wc)
        self.prev_output = 0.0
    
    def apply(self, input_signal: float) -> float:
        output = self.alpha * input_signal + (1.0 - self.alpha) * self.prev_output
        self.prev_output = output
        return output
    
    def reset(self):
        self.prev_output = 0.0


class SmartAdaptiveESC:
    """
    Smart Adaptive Extremum Seeking Controller.
    
    Automatically adapts perturbation magnitude and integrator gain based on:
    - Gradient magnitude (how steep the landscape is)
    - Setpoint stability (how much the setpoint changes)
    - Convergence detection (are we near the optimum?)
    - Reward variance (is the system noisy?)
    """
    
    def __init__(
        self,
        sampling_period: float,
        initial_dither_frequency: float,
        initial_dither_magnitude: float,
        initial_integrator_gain: float,
        initial_value: float,
        high_pass_cutoff: float,
        low_pass_cutoff: float,
        min_output: float = 0.0,
        max_output: float = 3.0,
        # Adaptive parameters
        enable_adaptive_dither: bool = True,
        enable_adaptive_gain: bool = True,
        enable_state_machine: bool = True,
        convergence_patience: int = 5,
        exploration_steps: int = 10,
    ):
        """
        Args:
            sampling_period: Time between updates (seconds)
            initial_dither_frequency: Starting perturbation frequency (rad/s)
            initial_dither_magnitude: Starting perturbation amplitude
            initial_integrator_gain: Starting gain (negative for descent)
            initial_value: Starting parameter value
            high_pass_cutoff: High-pass filter cutoff (rad/s)
            low_pass_cutoff: Low-pass filter cutoff (rad/s)
            min_output: Minimum allowed output value
            max_output: Maximum allowed output value
            enable_adaptive_dither: Whether to adapt dither magnitude
            enable_adaptive_gain: Whether to adapt integrator gain
            enable_state_machine: Whether to use state machine for convergence
            convergence_patience: Steps before declaring convergence
            exploration_steps: Initial exploration steps with high perturbation
        """
        # Basic parameters
        self.dt = sampling_period
        self.omega = initial_dither_frequency
        self.theta_0 = initial_value
        self.min_output = min_output
        self.max_output = max_output
        
        # Adaptive control flags
        self.enable_adaptive_dither = enable_adaptive_dither
        self.enable_adaptive_gain = enable_adaptive_gain
        self.enable_state_machine = enable_state_machine
        
        # Initial parameter ranges
        self.dither_min = initial_dither_magnitude * 0.1  # 10% of initial
        self.dither_max = initial_dither_magnitude * 2.0  # 200% of initial
        self.dither_current = initial_dither_magnitude
        
        self.gain_min = initial_integrator_gain * 0.1  # Conservative gain
        self.gain_max = initial_integrator_gain * 10.0  # Aggressive gain
        self.gain_current = initial_integrator_gain
        
        # Initialize filters
        self.hpf = HighPassFilter(sampling_period, high_pass_cutoff)
        self.lpf = LowPassFilter(sampling_period, low_pass_cutoff)
        
        # State variables
        self.phase = 0.0
        self.integral = 0.0
        self.prev_setpoint = initial_value
        
        # Gradient estimation
        self.m2 = 0.0  # Second moment for RMS
        self.beta = 0.9  # EMA coefficient for gradient
        self.epsilon = 1e-8
        
        # Convergence tracking
        self.state = ConvergenceState.EXPLORATION
        self.step_count = 0
        self.convergence_patience = convergence_patience
        self.exploration_steps = exploration_steps
        self.stable_steps = 0
        
        # History for convergence detection
        self.gradient_history = []
        self.setpoint_history = []
        self.reward_history = []
        self.history_length = 10
        
        # Setpoint rate limiting
        self.max_setpoint_change = 0.2
        
        # Performance metrics
        self.best_reward = -np.inf
        self.best_setpoint = initial_value
        self.steps_since_improvement = 0

    def _update_convergence_state(self, gradient_magnitude: float, reward: float):
        """Determine current convergence state using state machine."""
        if not self.enable_state_machine:
            return
        
        self.step_count += 1
        
        # Update history
        self.gradient_history.append(gradient_magnitude)
        self.setpoint_history.append(self.prev_setpoint)
        self.reward_history.append(reward)
        
        if len(self.gradient_history) > self.history_length:
            self.gradient_history.pop(0)
            self.setpoint_history.pop(0)
            self.reward_history.pop(0)
        
        # State machine transitions
        if self.state == ConvergenceState.EXPLORATION:
            # Stay in exploration for initial steps
            if self.step_count >= self.exploration_steps:
                self.state = ConvergenceState.SEEKING
                print(f"[ESC] State: EXPLORATION → SEEKING")
        
        elif self.state == ConvergenceState.SEEKING:
            # Check for convergence
            if len(self.gradient_history) >= self.convergence_patience:
                avg_grad = np.mean(self.gradient_history[-self.convergence_patience:])
                setpoint_var = np.var(self.setpoint_history[-self.convergence_patience:])
                
                # Converged if gradient is small and setpoint is stable
                if avg_grad < 0.05 and setpoint_var < 0.01:
                    self.stable_steps += 1
                    if self.stable_steps >= self.convergence_patience:
                        self.state = ConvergenceState.CONVERGED
                        print(f"[ESC] State: SEEKING → CONVERGED (grad={avg_grad:.4f}, var={setpoint_var:.4f})")
                else:
                    self.stable_steps = 0
        
        elif self.state == ConvergenceState.CONVERGED:
            # Check if we need to re-explore (reward dropped significantly)
            if len(self.reward_history) >= 3:
                recent_rewards = self.reward_history[-3:]
                if max(recent_rewards) < self.best_reward * 0.9:  # 10% drop
                    self.state = ConvergenceState.SEEKING
                    self.stable_steps = 0
                    print(f"[ESC] State: CONVERGED → SEEKING (reward dropped)")
            
            # Check for stable tracking
            if len(self.gradient_history) >= self.history_length:
                avg_grad = np.mean(self.gradient_history)
                if avg_grad < 0.01:
                    self.state = ConvergenceState.TRACKING
                    print(f"[ESC] State: CONVERGED → TRACKING")
        
        elif self.state == ConvergenceState.TRACKING:
            # Stay in tracking unless gradient increases significantly
            if gradient_magnitude > 0.1:
                self.state = ConvergenceState.SEEKING
                self.stable_steps = 0
                print(f"[ESC] State: TRACKING → SEEKING (large gradient)")
        
        # Update best reward
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_setpoint = self.prev_setpoint
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1

    def _compute_adaptive_dither(self, gradient_magnitude: float) -> float:
        """Compute adaptive dither magnitude based on state and gradient."""
        if not self.enable_adaptive_dither:
            return self.dither_current
        
        # State-based dither adjustment
        state_multipliers = {
            ConvergenceState.EXPLORATION: 1.5,   # Large for exploration
            ConvergenceState.SEEKING: 1.0,       # Normal for seeking
            ConvergenceState.CONVERGED: 0.3,     # Small near optimum
            ConvergenceState.TRACKING: 0.1,      # Minimal for tracking
        }
        
        base_dither = self.dither_current * state_multipliers[self.state]
        
        # Gradient-based adjustment (larger gradient = reduce dither)
        if gradient_magnitude > 0.2:
            gradient_factor = 0.7  # Reduce dither in steep regions
        elif gradient_magnitude < 0.05:
            gradient_factor = 1.2  # Increase dither in flat regions
        else:
            gradient_factor = 1.0
        
        # Compute new dither
        new_dither = base_dither * gradient_factor
        new_dither = np.clip(new_dither, self.dither_min, self.dither_max)
        
        # Smooth transition (exponential moving average)
        alpha = 0.1  # Slow adaptation
        self.dither_current = alpha * new_dither + (1 - alpha) * self.dither_current
        
        return self.dither_current

    def _compute_adaptive_gain(self, gradient_magnitude: float) -> float:
        """Compute adaptive integrator gain based on state and gradient."""
        if not self.enable_adaptive_gain:
            return self.gain_current
        
        # State-based gain adjustment
        state_multipliers = {
            ConvergenceState.EXPLORATION: 2.0,   # Fast exploration
            ConvergenceState.SEEKING: 1.0,       # Normal seeking
            ConvergenceState.CONVERGED: 0.3,     # Slow refinement
            ConvergenceState.TRACKING: 0.1,      # Minimal tracking
        }
        
        base_gain = self.gain_current * state_multipliers[self.state]
        
        # Gradient-based adjustment (larger gradient = higher gain, but saturate)
        if gradient_magnitude > 0.5:
            gradient_factor = 2.0  # High gain in steep regions
        elif gradient_magnitude > 0.2:
            gradient_factor = 1.5
        elif gradient_magnitude < 0.05:
            gradient_factor = 0.5  # Low gain in flat regions
        else:
            gradient_factor = 1.0
        
        # Compute new gain
        new_gain = base_gain * gradient_factor
        new_gain = np.clip(new_gain, self.gain_max, self.gain_min)  # Note: negative values
        
        # Smooth transition
        alpha = 0.2
        self.gain_current = alpha * new_gain + (1 - alpha) * self.gain_current
        
        return self.gain_current

    def _apply_rate_limiting(self, setpoint: float) -> float:
        """Apply rate limiting to prevent large jumps."""
        delta = setpoint - self.prev_setpoint
        
        # Adaptive rate limit based on state
        state_rate_limits = {
            ConvergenceState.EXPLORATION: self.max_setpoint_change * 2.0,
            ConvergenceState.SEEKING: self.max_setpoint_change,
            ConvergenceState.CONVERGED: self.max_setpoint_change * 0.5,
            ConvergenceState.TRACKING: self.max_setpoint_change * 0.2,
        }
        
        rate_limit = state_rate_limits[self.state]
        
        if abs(delta) > rate_limit:
            setpoint = self.prev_setpoint + np.sign(delta) * rate_limit
            # Anti-windup: adjust integral to match rate-limited setpoint
            self.integral = setpoint - self.theta_0
        
        return setpoint

    def update(self, cost: float) -> Tuple[float, float, float, float, float, float, dict]:
        """
        Update the controller with a new cost measurement.
        
        Args:
            cost: Current value of the cost function (negative reward)
            
        Returns:
            Tuple containing:
                - output: Perturbed parameter value
                - hpf_output: High-pass filter output
                - lpf_output: Low-pass filter output (gradient estimate)
                - gradient_magnitude: RMS of gradient estimate
                - gradient: Raw gradient estimate
                - setpoint: Current setpoint (without perturbation)
                - info: Dictionary with adaptive control info
        """
        # 1. High-pass filter to remove DC component
        hpf_output = self.hpf.apply(cost)
        
        # 2. Demodulate by multiplying with sin(wt)
        demodulated = hpf_output * np.sin(self.phase)
        
        # 3. Low-pass filter to extract gradient estimate
        lpf_output = self.lpf.apply(demodulated)
        
        # 4. Compute gradient magnitude
        self.m2 = self.beta * self.m2 + (1.0 - self.beta) * (lpf_output ** 2)
        gradient_magnitude = np.sqrt(self.m2 + self.epsilon)
        
        # 5. Update convergence state
        reward = -cost  # Convert cost back to reward for state tracking
        self._update_convergence_state(gradient_magnitude, reward)
        
        # 6. Compute adaptive parameters
        adaptive_dither = self._compute_adaptive_dither(gradient_magnitude)
        adaptive_gain = self._compute_adaptive_gain(gradient_magnitude)
        
        # 7. Integrate with adaptive gain
        self.integral += adaptive_gain * lpf_output * self.dt
        
        # 8. Compute raw setpoint
        setpoint_raw = self.theta_0 + self.integral
        
        # 9. Apply constraints with improved anti-windup
        setpoint = np.clip(setpoint_raw, self.min_output, self.max_output)
        
        # Anti-windup: back-calculate integral if saturated
        if setpoint != setpoint_raw:
            # Only prevent further integration in wrong direction
            if setpoint == self.min_output and adaptive_gain * lpf_output < 0:
                self.integral = self.min_output - self.theta_0
            elif setpoint == self.max_output and adaptive_gain * lpf_output > 0:
                self.integral = self.max_output - self.theta_0
        
        # 10. Apply rate limiting
        setpoint = self._apply_rate_limiting(setpoint)
        
        # 11. Add adaptive perturbation
        perturbation = adaptive_dither * np.sin(self.phase)
        output = np.clip(setpoint + perturbation, self.min_output, self.max_output)
        
        # 12. Update phase
        self.phase += self.omega * self.dt
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi
        
        # 13. Update previous setpoint
        self.prev_setpoint = setpoint
        
        # 14. Compile adaptive control info
        info = {
            "state": self.state.value,
            "adaptive_dither": adaptive_dither,
            "adaptive_gain": adaptive_gain,
            "dither_reduction": (self.dither_max - adaptive_dither) / self.dither_max,
            "steps_in_state": self.step_count,
            "best_reward": self.best_reward,
            "best_setpoint": self.best_setpoint,
            "steps_since_improvement": self.steps_since_improvement,
        }
        
        return (
            output,
            hpf_output,
            lpf_output,
            gradient_magnitude,
            lpf_output,
            setpoint,
            info
        )
    
    def reset(self):
        """Reset controller state to initial conditions."""
        self.hpf.reset()
        self.lpf.reset()
        self.phase = 0.0
        self.integral = 0.0
        self.m2 = 0.0
        self.state = ConvergenceState.EXPLORATION
        self.step_count = 0
        self.stable_steps = 0
        self.gradient_history.clear()
        self.setpoint_history.clear()
        self.reward_history.clear()
        self.best_reward = -np.inf
        self.steps_since_improvement = 0
    
    def get_state(self) -> dict:
        """Get comprehensive controller state."""
        return {
            "phase": self.phase,
            "integral": self.integral,
            "m2": self.m2,
            "gradient_magnitude": np.sqrt(self.m2 + self.epsilon),
            "convergence_state": self.state.value,
            "dither_current": self.dither_current,
            "gain_current": self.gain_current,
            "best_reward": self.best_reward,
            "best_setpoint": self.best_setpoint,
        }