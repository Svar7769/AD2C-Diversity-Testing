"""
Improved Adaptive Extremum Seeking Control (ESC) implementation.
A gradient-free optimization method that uses sinusoidal perturbations
to estimate gradients and converge to optimal parameter values.

Key improvements over previous adaptive version:
1. Corrected gain adaptation logic (increase gain when gradient is large)
2. Maintains persistent excitation (no dither decay)
3. Simplified adaptive mechanisms
4. Better noise handling
"""
import numpy as np
from typing import Tuple


class HighPassFilter:
    """First-order high-pass filter to remove DC offset from signals."""
    
    def __init__(self, sampling_period: float, cutoff_frequency: float):
        """
        Args:
            sampling_period: Time between samples (seconds)
            cutoff_frequency: Cutoff frequency in rad/s
        """
        self.dt = sampling_period
        self.wc = cutoff_frequency
        
        # Filter coefficient: alpha = wc / (wc + 1/dt)
        self.alpha = self.wc / (self.wc + 1.0 / self.dt)
        
        # Previous values for filtering
        self.prev_input = 0.0
        self.prev_output = 0.0
    
    def apply(self, input_signal: float) -> float:
        """Apply high-pass filter to input signal."""
        # HPF formula: y[k] = alpha * (y[k-1] + x[k] - x[k-1])
        output = self.alpha * (self.prev_output + input_signal - self.prev_input)
        
        self.prev_input = input_signal
        self.prev_output = output
        
        return output
    
    def reset(self):
        """Reset filter state."""
        self.prev_input = 0.0
        self.prev_output = 0.0


class LowPassFilter:
    """First-order low-pass filter to smooth signals."""
    
    def __init__(self, sampling_period: float, cutoff_frequency: float):
        """
        Args:
            sampling_period: Time between samples (seconds)
            cutoff_frequency: Cutoff frequency in rad/s
        """
        self.dt = sampling_period
        self.wc = cutoff_frequency
        
        # Filter coefficient: alpha = dt * wc / (1 + dt * wc)
        self.alpha = (self.dt * self.wc) / (1.0 + self.dt * self.wc)
        
        # Previous output for filtering
        self.prev_output = 0.0
    
    def apply(self, input_signal: float) -> float:
        """Apply low-pass filter to input signal."""
        # LPF formula: y[k] = alpha * x[k] + (1 - alpha) * y[k-1]
        output = self.alpha * input_signal + (1.0 - self.alpha) * self.prev_output
        
        self.prev_output = output
        
        return output
    
    def reset(self):
        """Reset filter state."""
        self.prev_output = 0.0


class AdaptiveExtremumSeekingController:
    """
    Improved Adaptive Extremum Seeking Controller.
    
    Key features:
    - Adaptive gain that increases when far from optimum (corrected logic)
    - Maintains constant dither for persistent excitation
    - Optional noise-adaptive gain scaling
    - Simplified implementation with better stability
    """
    
    def __init__(
        self,
        sampling_period: float,
        dither_frequency: float,
        dither_magnitude: float,
        integrator_gain: float,
        initial_value: float,
        high_pass_cutoff: float,
        low_pass_cutoff: float,
        use_adaptive_gain: bool = True,
        adaptive_gain_range: float = 2.5,  # Max multiplier for adaptive gain
        gradient_threshold: float = 0.2,   # Threshold for high/low gradient
        min_output: float = 0.0
    ):
        """
        Args:
            sampling_period: Time between updates (seconds)
            dither_frequency: Perturbation frequency (rad/s)
            dither_magnitude: Amplitude of perturbation (kept constant)
            integrator_gain: Base gain for parameter updates (negative for gradient descent)
            initial_value: Starting parameter value
            high_pass_cutoff: High-pass filter cutoff (rad/s)
            low_pass_cutoff: Low-pass filter cutoff (rad/s)
            use_adaptive_gain: Whether to use adaptive gain adjustment
            adaptive_gain_range: Maximum multiplier for gain (e.g., 2.5 = up to 2.5x base gain)
            gradient_threshold: Gradient RMS threshold for switching behavior
            min_output: Minimum allowed output value
        """
        # Validate frequency ordering for stability
        if high_pass_cutoff >= low_pass_cutoff:
            raise ValueError(
                f"ESC requires high_pass_cutoff < low_pass_cutoff for stability. "
                f"Got: ωh={high_pass_cutoff:.3f} >= ωl={low_pass_cutoff:.3f}. "
                f"Classical theory requires: ωh << ωl << ω."
            )
        if low_pass_cutoff >= dither_frequency:
            raise ValueError(
                f"ESC requires low_pass_cutoff < dither_frequency for stability. "
                f"Got: ωl={low_pass_cutoff:.3f} >= ω={dither_frequency:.3f}. "
                f"Classical theory requires: ωh << ωl << ω."
            )
        
        self.dt = sampling_period
        self.omega = dither_frequency
        self.a = dither_magnitude  # Keep constant for persistent excitation
        self.k = integrator_gain  # Base gain (negative for descent)
        self.theta_0 = initial_value
        self.min_output = min_output
        
        # Adaptive parameters
        self.use_adaptive = use_adaptive_gain
        self.adaptive_range = adaptive_gain_range
        self.gradient_threshold = gradient_threshold
        
        # Compute high gain (used when gradient is large = far from optimum)
        self.high_gain = abs(integrator_gain) * adaptive_gain_range
        if integrator_gain < 0:  # Maintain sign convention
            self.high_gain = -self.high_gain
        
        # Initialize filters
        self.hpf = HighPassFilter(sampling_period, high_pass_cutoff)
        self.lpf = LowPassFilter(sampling_period, low_pass_cutoff)
        
        # State variables
        self.phase = 0.0
        self.integral = 0.0
        
        # Gradient statistics
        self.m2 = 0.0  # Second moment for RMS calculation
        self.beta = 0.8  # EMA coefficient for gradient RMS
        self.epsilon = 1e-8
        
        # Tracking for logging
        self.current_gain = integrator_gain
    
    def update(self, cost: float) -> Tuple[float, float, float, float, float, float]:
        """
        Update the controller with a new cost measurement.
        
        Args:
            cost: Current value of the cost function
            
        Returns:
            Tuple containing:
                - output: Perturbed parameter value (setpoint + dither)
                - hpf_output: High-pass filter output
                - lpf_output: Low-pass filter output (gradient estimate)
                - gradient_magnitude: RMS of gradient estimate
                - gradient: Raw gradient estimate
                - setpoint: Current setpoint (without perturbation)
        """
        # 1. High-pass filter to remove DC component
        hpf_output = self.hpf.apply(cost)
        
        # 2. Demodulate by multiplying with sin(ωt)
        demodulated = hpf_output * np.sin(self.phase)
        
        # 3. Low-pass filter to extract gradient estimate
        lpf_output = self.lpf.apply(demodulated)
        
        # 4. Update gradient RMS using exponential moving average
        self.m2 = self.beta * self.m2 + (1.0 - self.beta) * (lpf_output ** 2)
        gradient_magnitude = np.sqrt(self.m2 + self.epsilon)
        
        # 5. Compute adaptive gain (CORRECTED LOGIC)
        if self.use_adaptive:
            # Use HIGH gain when gradient is LARGE (far from optimum)
            # Use BASE gain when gradient is SMALL (near optimum)
            if gradient_magnitude > self.gradient_threshold:
                gain = self.high_gain  # Aggressive updates when far away
            else:
                gain = self.k  # Conservative updates when close
        else:
            gain = self.k
        
        self.current_gain = gain  # Track for logging
        
        # 6. Integrate gradient to update parameter estimate
        self.integral += gain * lpf_output * self.dt
        
        # 7. Compute setpoint (base parameter value)
        setpoint_raw = self.theta_0 + self.integral
        
        # 8. Apply output constraints with anti-windup
        setpoint = max(setpoint_raw, self.min_output)
        
        # 9. Anti-windup correction
        if setpoint_raw < self.min_output:
            self.integral = self.min_output - self.theta_0
        
        # 10. Add constant perturbation (persistent excitation)
        perturbation = self.a * np.sin(self.phase)
        output = setpoint + perturbation
        
        # 11. Update phase
        self.phase += self.omega * self.dt
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi
        
        return (
            output,
            hpf_output,
            lpf_output,
            gradient_magnitude,
            lpf_output,  # Raw gradient (same as lpf_output)
            setpoint
        )
    
    def reset(self):
        """Reset controller state to initial conditions."""
        self.hpf.reset()
        self.lpf.reset()
        self.phase = 0.0
        self.integral = 0.0
        self.m2 = 0.0
        self.current_gain = self.k
    
    def get_state(self) -> dict:
        """Get current controller state (similar logging to old ESC)."""
        return {
            "phase": self.phase,
            "integral": self.integral,
            "m2": self.m2,
            "gradient_magnitude": np.sqrt(self.m2),
            "current_gain": self.current_gain,
            "gain_multiplier": abs(self.current_gain / self.k) if self.k != 0 else 1.0,
        }


class ExtremumSeekingController(AdaptiveExtremumSeekingController):
    """
    Alias for backward compatibility with original ESC implementation.
    This is just ImprovedAdaptiveESC with adaptive features disabled by default.
    """
    
    def __init__(
        self,
        sampling_period: float,
        dither_frequency: float,
        dither_magnitude: float,
        integrator_gain: float,
        initial_value: float,
        high_pass_cutoff: float,
        low_pass_cutoff: float,
        use_adaptive_gain: bool = True,
        min_output: float = 0.0
    ):
        """Classic ESC constructor signature for backward compatibility."""
        super().__init__(
            sampling_period=sampling_period,
            dither_frequency=dither_frequency,
            dither_magnitude=dither_magnitude,
            integrator_gain=integrator_gain,
            initial_value=initial_value,
            high_pass_cutoff=high_pass_cutoff,
            low_pass_cutoff=low_pass_cutoff,
            use_adaptive_gain=use_adaptive_gain,
            adaptive_gain_range=2.5,  # Default from old implementation
            gradient_threshold=0.2,   # Default from old implementation
            min_output=min_output
        )