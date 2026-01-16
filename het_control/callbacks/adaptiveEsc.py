"""
Extremum Seeking Control (ESC) implementation with adaptive extensions.
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


class AdaptiveExtremumSeekingController:
    """
    Extremum Seeking Controller for real-time optimization with adaptive extensions.
    
    Uses sinusoidal perturbations to estimate gradients of an unknown cost function
    and adjusts parameters to find the optimum. Supports adaptive gain and dither
    magnitude based on RMSprop-style gradient normalization.
    
    When adaptive features are disabled, behaves as classical ESC.
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
        use_adaptive_gain: bool = False,
        use_adaptive_dither: bool = False,
        min_output: float = 0.0,
        # Adaptive gain parameters
        gain_adaptation_mode: str = "rmsprop",  # "rmsprop", "binary", or "gradient_norm"
        binary_gain_threshold: float = 0.2,
        binary_high_gain_multiplier: float = 2.5,
        # Adaptive dither parameters
        dither_decay_rate: float = 0.999,
        min_dither_ratio: float = 0.1,
        dither_boost_threshold: float = 0.01,
        dither_boost_rate: float = 1.02,
    ):
        """
        Args:
            sampling_period: Time between updates (seconds)
            dither_frequency: Perturbation frequency (rad/s)
            dither_magnitude: Initial amplitude of perturbation
            integrator_gain: Base gain for parameter updates (negative for gradient descent)
            initial_value: Starting parameter value
            high_pass_cutoff: High-pass filter cutoff (rad/s)
            low_pass_cutoff: Low-pass filter cutoff (rad/s)
            use_adaptive_gain: Whether to use adaptive gain adjustment
            use_adaptive_dither: Whether to use adaptive dither magnitude
            min_output: Minimum allowed output value
            
            # Adaptive Gain Parameters:
            gain_adaptation_mode: Mode for adaptive gain
                - "rmsprop": RMSprop-style inverse square root scaling (recommended)
                - "binary": Binary switching between two gain values (legacy)
                - "gradient_norm": Smooth scaling based on gradient magnitude
            binary_gain_threshold: Threshold for binary mode switching
            binary_high_gain_multiplier: Multiplier for high gain in binary mode
            
            # Adaptive Dither Parameters:
            dither_decay_rate: Rate at which dither decays (0.995-0.999, closer to 1 = slower)
            min_dither_ratio: Minimum dither as fraction of initial (e.g., 0.1 = 10%)
            dither_boost_threshold: Gradient magnitude below which to boost dither
            dither_boost_rate: Rate at which dither increases when boosted (e.g., 1.02 = 2% increase)
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
        self.omega = dither_frequency  # Perturbation frequency
        self.a_initial = dither_magnitude  # Initial perturbation amplitude
        self.a = dither_magnitude  # Current perturbation amplitude (adaptive)
        self.k = integrator_gain  # Base integrator gain (negative for descent)
        self.theta_0 = initial_value  # Initial setpoint
        self.min_output = min_output
        
        # Adaptive feature flags
        self.use_adaptive_gain = use_adaptive_gain
        self.use_adaptive_dither = use_adaptive_dither
        
        # Adaptive gain configuration
        self.gain_adaptation_mode = gain_adaptation_mode
        self.binary_gain_threshold = binary_gain_threshold
        self.binary_high_gain = abs(integrator_gain) * binary_high_gain_multiplier
        if integrator_gain < 0:  # Maintain sign convention
            self.binary_high_gain = -self.binary_high_gain
        
        # Adaptive dither configuration
        self.dither_decay_rate = dither_decay_rate
        self.min_dither = dither_magnitude * min_dither_ratio
        self.dither_boost_threshold = dither_boost_threshold
        self.dither_boost_rate = dither_boost_rate
        
        # Initialize filters
        self.hpf = HighPassFilter(sampling_period, high_pass_cutoff)
        self.lpf = LowPassFilter(sampling_period, low_pass_cutoff)
        
        # State variables
        self.phase = 0.0  # Current phase of perturbation (ωt)
        self.integral = 0.0  # Integrator state
        
        # Gradient statistics (for adaptive features)
        self.m2 = 0.0  # Second moment estimate (for gradient RMS)
        self.v2 = 0.0  # Second moment of HPF output (for cost variance)
        self.beta = 0.9  # Exponential moving average coefficient
        self.epsilon = 1e-8  # Small constant to prevent division by zero
        
        # Tracking for logging
        self.current_gain = integrator_gain  # Track actual gain used
        self.stuck_counter = 0  # Count steps with low gradient
    
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
        
        # 2. Demodulate by multiplying with sin(ωt)
        demodulated = hpf_output * np.sin(self.phase)
        
        # 3. Low-pass filter to extract gradient estimate
        lpf_output = self.lpf.apply(demodulated)
        
        # 4. Update gradient magnitude (RMS) using exponential moving average
        self.m2 = self.beta * self.m2 + (1.0 - self.beta) * (lpf_output ** 2)
        gradient_rms = np.sqrt(self.m2 + self.epsilon)
        
        # 5. Update cost variance (for dither adaptation)
        self.v2 = self.beta * self.v2 + (1.0 - self.beta) * (hpf_output ** 2)
        cost_variance = np.sqrt(self.v2 + self.epsilon)
        
        # 6. Compute adaptive gain
        gain = self._compute_adaptive_gain(lpf_output, gradient_rms)
        self.current_gain = gain  # Track for logging
        
        # 7. Integrate gradient to update parameter estimate
        self.integral += gain * lpf_output * self.dt
        
        # 8. Compute setpoint (base parameter value without perturbation)
        setpoint_raw = self.theta_0 + self.integral
        
        # 9. Apply output constraints (clamping with anti-windup)
        setpoint = max(setpoint_raw, self.min_output)
        
        # 10. Anti-windup: correct integrator if output is saturated
        if setpoint_raw < self.min_output:
            self.integral = self.min_output - self.theta_0
        
        # 11. Update adaptive dither magnitude
        self._update_adaptive_dither(gradient_rms, cost_variance)
        
        # 12. Add perturbation to get final output
        perturbation = self.a * np.sin(self.phase)
        output = setpoint + perturbation
        
        # 13. Update phase for next iteration
        self.phase += self.omega * self.dt
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi
        
        return (
            output,           # Total output (setpoint + perturbation)
            hpf_output,       # High-pass filtered cost
            lpf_output,       # Gradient estimate
            gradient_rms,     # RMS of gradient
            lpf_output,       # Raw gradient (same as lpf_output)
            setpoint          # Base setpoint (no perturbation)
        )
    
    def _compute_adaptive_gain(self, gradient: float, gradient_rms: float) -> float:
        """
        Compute adaptive gain based on gradient statistics.
        
        Args:
            gradient: Current gradient estimate
            gradient_rms: RMS of gradient over time
            
        Returns:
            Adaptive gain value
        """
        if not self.use_adaptive_gain:
            # Classical ESC: fixed gain
            return self.k
        
        if self.gain_adaptation_mode == "rmsprop":
            # RMSprop-style: scale inversely with gradient RMS
            # When gradients are large → smaller steps
            # When gradients are small → larger steps (but limited by base gain)
            adaptive_gain = self.k / (gradient_rms / abs(self.k) + 1.0)
            return adaptive_gain
            
        elif self.gain_adaptation_mode == "binary":
            # Binary switching (legacy mode for comparison)
            # Use high gain when gradient is large (far from optimum)
            if gradient_rms > self.binary_gain_threshold:
                return self.binary_high_gain
            else:
                return self.k
                
        elif self.gain_adaptation_mode == "gradient_norm":
            # Smooth scaling based on normalized gradient magnitude
            # gain = k * (1 + gradient_rms) - increases with gradient
            scale = 1.0 / (1.0 + gradient_rms / self.binary_gain_threshold)
            adaptive_gain = self.k * (0.5 + 0.5 * scale)  # Range: [0.5k, k]
            return adaptive_gain
        
        else:
            raise ValueError(f"Unknown gain_adaptation_mode: {self.gain_adaptation_mode}")
    
    def _update_adaptive_dither(self, gradient_rms: float, cost_variance: float):
        """
        Update dither magnitude adaptively based on optimization progress.
        
        Implements exploration-exploitation tradeoff:
        - Decay dither over time (exploitation)
        - Boost dither when stuck in flat regions (exploration)
        
        Args:
            gradient_rms: RMS of gradient estimate
            cost_variance: Variance of cost signal
        """
        if not self.use_adaptive_dither:
            # Classical ESC: fixed dither
            return
        
        # Decay dither over time (exploration → exploitation)
        self.a *= self.dither_decay_rate
        
        # Enforce minimum dither to maintain exploration
        self.a = max(self.a, self.min_dither)
        
        # Detect if stuck: low gradient with high cost variance
        # This suggests we're in a flat region or noisy area
        if gradient_rms < self.dither_boost_threshold and cost_variance > 0.05:
            self.stuck_counter += 1
            # Boost dither if stuck for multiple steps
            if self.stuck_counter > 3:
                self.a = min(self.a * self.dither_boost_rate, self.a_initial)
        else:
            self.stuck_counter = 0
    
    def reset(self):
        """Reset controller state to initial conditions."""
        self.hpf.reset()
        self.lpf.reset()
        self.phase = 0.0
        self.integral = 0.0
        self.m2 = 0.0
        self.v2 = 0.0
        self.a = self.a_initial  # Reset dither to initial value
        self.stuck_counter = 0
        self.current_gain = self.k
    
    def get_state(self) -> dict:
        """Get current controller state."""
        return {
            "phase": self.phase,
            "integral": self.integral,
            "m2": self.m2,
            "v2": self.v2,
            "gradient_magnitude": np.sqrt(self.m2),
            "cost_variance": np.sqrt(self.v2),
            "dither_amplitude": self.a,
            "dither_ratio": self.a / self.a_initial,
            "current_gain": self.current_gain,
            "stuck_counter": self.stuck_counter,
        }