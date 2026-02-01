"""
Extremum Seeking Control (ESC) implementation for REWARD MAXIMIZATION.
Uses RMSprop adaptive learning rate and vanishing perturbation.
"""
import numpy as np
from typing import Dict, Any


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
    Extremum Seeking Controller for REWARD MAXIMIZATION.

    Features:
    - RMSprop adaptive learning rate (bias-corrected)
    - Vanishing perturbation based on gradient magnitude
    - Gradient ASCENT (positive sign) for reward maximization
    """

    def __init__(
        self,
        sampling_period: float,
        dither_frequency: float,
        dither_amplitude: float,
        learning_rate: float,
        initial_value: float,
        high_pass_cutoff: float,
        low_pass_cutoff: float,
        min_output: float = 0.0,
        max_output: float = 3.0,
        # RMSprop parameters
        use_rmsprop: bool = True,
        beta: float = 0.9,
        epsilon: float = 1e-8,
        max_lr_multiplier: float = 10.0,
        # Vanishing perturbation
        use_vanishing_perturbation: bool = True,
        min_perturbation_ratio: float = 0.1,
    ):
        """
        Args:
            sampling_period: Time between updates (seconds)
            dither_frequency: Perturbation frequency (rad/s)
            dither_amplitude: Initial perturbation amplitude
            learning_rate: Base learning rate for gradient ascent
            initial_value: Starting parameter value
            high_pass_cutoff: High-pass filter cutoff (rad/s)
            low_pass_cutoff: Low-pass filter cutoff (rad/s)
            min_output: Minimum allowed output value
            max_output: Maximum allowed output value
            use_rmsprop: Whether to use RMSprop adaptive learning rate
            beta: RMSprop decay factor (typically 0.9)
            epsilon: Small constant for numerical stability
            max_lr_multiplier: Maximum learning rate as multiple of base
            use_vanishing_perturbation: Whether to reduce perturbation as gradient decreases
            min_perturbation_ratio: Minimum perturbation as ratio of initial
        """
        # Basic parameters
        self.dt = sampling_period
        self.omega = dither_frequency
        self.theta_0 = initial_value
        self.min_output = min_output
        self.max_output = max_output

        # Learning rate
        self.k = learning_rate
        self.max_lr = learning_rate * max_lr_multiplier

        # RMSprop
        self.use_rmsprop = use_rmsprop
        self.beta = beta
        self.epsilon = epsilon
        self.v = 0.0  # Second moment estimate

        # Perturbation parameters
        self.use_vanishing_perturbation = use_vanishing_perturbation
        self.a_initial = dither_amplitude
        self.a = dither_amplitude
        self.min_perturbation = dither_amplitude * min_perturbation_ratio

        # Initialize filters
        self.hpf = HighPassFilter(sampling_period, high_pass_cutoff)
        self.lpf = LowPassFilter(sampling_period, low_pass_cutoff)

        # State variables
        self.phase = 0.0
        self.integral = 0.0
        self.prev_setpoint = initial_value
        self.step_count = 0

    def update(self, reward: float) -> Dict[str, Any]:
        """
        Update controller with reward signal (MAXIMIZATION).

        Args:
            reward: Current reward value (higher = better)

        Returns:
            Dictionary containing:
                - output: Perturbed parameter value
                - setpoint: Current setpoint (without perturbation)
                - gradient: Gradient estimate
                - gradient_magnitude: RMS of gradient estimate
                - perturbation_amplitude: Current dither amplitude
                - adaptive_lr: Current learning rate
                - hpf_output: High-pass filter output
        """
        self.step_count += 1

        # 1. High-pass filter removes DC offset (unknown baseline reward)
        hpf_output = self.hpf.apply(reward)

        # 2. Demodulate by multiplying with sin(wt)
        demodulated = hpf_output * np.sin(self.phase)

        # 3. Low-pass filter to extract gradient estimate
        gradient = self.lpf.apply(demodulated)

        # 4. RMSprop adaptive learning rate
        if self.use_rmsprop:
            self.v = self.beta * self.v + (1.0 - self.beta) * (gradient ** 2)
            # Bias correction
            v_corrected = self.v / (1.0 - self.beta ** min(self.step_count, 1000))
            raw_adaptive_lr = self.k / (np.sqrt(v_corrected) + self.epsilon)
            adaptive_lr = min(raw_adaptive_lr, self.max_lr)
        else:
            v_corrected = gradient ** 2
            adaptive_lr = self.k

        gradient_magnitude = np.sqrt(v_corrected + self.epsilon)

        # 5. GRADIENT ASCENT (positive sign for maximization!)
        self.integral += adaptive_lr * gradient * self.dt

        # 6. Compute raw setpoint
        setpoint_raw = self.theta_0 + self.integral

        # 7. Apply constraints with anti-windup
        setpoint = np.clip(setpoint_raw, self.min_output, self.max_output)

        if setpoint_raw != setpoint:
            self.integral = setpoint - self.theta_0

        # 8. Compute adaptive perturbation (vanishing based on gradient magnitude)
        if self.use_vanishing_perturbation:
            scale = np.tanh(gradient_magnitude)
            self.a = self.min_perturbation + (self.a_initial - self.min_perturbation) * scale
        else:
            self.a = self.a_initial

        # 9. Compute output with perturbation
        perturbation = self.a * np.sin(self.phase)
        output = np.clip(setpoint + perturbation, self.min_output, self.max_output)

        # 10. Update phase
        self.phase += self.omega * self.dt
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi

        # 11. Update previous setpoint
        self.prev_setpoint = setpoint

        return {
            "output": output,
            "setpoint": setpoint,
            "gradient": gradient,
            "gradient_magnitude": gradient_magnitude,
            "perturbation_amplitude": self.a,
            "adaptive_lr": adaptive_lr,
            "hpf_output": hpf_output,
        }

    def reset(self):
        """Reset controller state to initial conditions."""
        self.hpf.reset()
        self.lpf.reset()
        self.phase = 0.0
        self.integral = 0.0
        self.v = 0.0
        self.a = self.a_initial
        self.step_count = 0
        self.prev_setpoint = self.theta_0

    def get_state(self) -> Dict[str, Any]:
        """Get controller state."""
        return {
            "phase": self.phase,
            "integral": self.integral,
            "v": self.v,
            "gradient_magnitude": np.sqrt(self.v / (1.0 - self.beta ** max(self.step_count, 1)) + self.epsilon) if self.step_count > 0 else 0.0,
            "dither_current": self.a,
            "step_count": self.step_count,
        }
