"""
Extremum Seeking Control (ESC) implementation.
A gradient-free optimization method that uses sinusoidal perturbations
to estimate gradients and converge to optimal parameter values.
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


class ExtremumSeekingController:
    """
    Extremum Seeking Controller for real-time optimization.
    
    Uses sinusoidal perturbations to estimate gradients of an unknown cost function
    and adjusts parameters to find the optimum.
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
        """
        Args:
            sampling_period: Time between updates (seconds)
            dither_frequency: Perturbation frequency (rad/s)
            dither_magnitude: Amplitude of perturbation
            integrator_gain: Base gain for parameter updates (negative for gradient descent)
            initial_value: Starting parameter value
            high_pass_cutoff: High-pass filter cutoff (rad/s)
            low_pass_cutoff: Low-pass filter cutoff (rad/s)
            use_adaptive_gain: Whether to use adaptive gain switching
            min_output: Minimum allowed output value
        """
        self.dt = sampling_period
        self.omega = dither_frequency  # Perturbation frequency
        self.a = dither_magnitude  # Perturbation amplitude
        self.k = integrator_gain  # Base integrator gain (should be negative for descent)
        self.theta_0 = initial_value  # Initial setpoint
        self.use_adaptive = use_adaptive_gain
        self.min_output = min_output
        
        # Initialize filters
        self.hpf = HighPassFilter(sampling_period, high_pass_cutoff)
        self.lpf = LowPassFilter(sampling_period, low_pass_cutoff)
        
        # State variables
        self.phase = 0.0  # Current phase of perturbation (wt)
        self.integral = 0.0  # Integrator state
        
        # Adaptive gain parameters
        self.m2 = 0.0  # Second moment estimate (for RMS)
        self.beta = 0.8  # Exponential moving average coefficient
        self.epsilon = 1e-8  # Small constant to prevent division by zero
        
        # Adaptive gain thresholds
        self.gradient_threshold = 0.2
        self.high_gain = -0.1  # Used when gradient magnitude is high
    
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
        # 1. High-pass filter to remove DC component from cost signal
        hpf_output = self.hpf.apply(cost)
        
        # 2. Demodulate by multiplying with sin(wt)
        demodulated = hpf_output * np.sin(self.phase)
        
        # 3. Low-pass filter to extract gradient estimate
        lpf_output = self.lpf.apply(demodulated)
        
        # 4. Compute gradient magnitude using exponential moving average of squared gradient
        self.m2 = self.beta * self.m2 + (1.0 - self.beta) * (lpf_output ** 2)
        gradient_magnitude = np.sqrt(self.m2)
        
        # 5. Determine integrator gain (adaptive or fixed)
        if self.use_adaptive:
            # Use high gain when gradient is large, base gain when gradient is small
            gain = self.high_gain if gradient_magnitude > self.gradient_threshold else self.k
        else:
            gain = self.k
        
        # 6. Integrate gradient to update parameter estimate
        self.integral += gain * lpf_output * self.dt
        
        # 7. Compute setpoint (base parameter value without perturbation)
        setpoint_raw = self.theta_0 + self.integral
        
        # 8. Apply output constraints (clamping with anti-windup)
        setpoint = max(setpoint_raw, self.min_output)
        
        # 9. Anti-windup: correct integrator if output is saturated
        if setpoint_raw < self.min_output:
            self.integral = self.min_output - self.theta_0
        
        # 10. Add perturbation to get final output
        perturbation = self.a * np.sin(self.phase)
        output = setpoint + perturbation
        
        # 11. Update phase for next iteration
        self.phase += self.omega * self.dt
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi
        
        return (
            output,           # Total output (setpoint + perturbation)
            hpf_output,       # High-pass filtered cost
            lpf_output,       # Gradient estimate
            gradient_magnitude,  # RMS of gradient
            lpf_output,       # Raw gradient (same as lpf_output)
            setpoint          # Base setpoint (no perturbation)
        )
    
    def reset(self):
        """Reset controller state to initial conditions."""
        self.hpf.reset()
        self.lpf.reset()
        self.phase = 0.0
        self.integral = 0.0
        self.m2 = 0.0
    
    def get_state(self) -> dict:
        """Get current controller state."""
        return {
            "phase": self.phase,
            "integral": self.integral,
            "m2": self.m2,
            "gradient_magnitude": np.sqrt(self.m2)
        }