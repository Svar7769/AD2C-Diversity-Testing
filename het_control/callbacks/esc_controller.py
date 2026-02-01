"""
Extremum Seeking Control (ESC) implementation.
A gradient-free optimization method that uses sinusoidal perturbations
to estimate gradients and converge to optimal parameter values.

Uses Tustin (bilinear) transform for filters and trapezoidal integration.
"""
import numpy as np
from typing import Tuple


class ExtremumSeekingController:
    """
    Extremum Seeking Controller for real-time optimization.

    Uses sinusoidal perturbations to estimate gradients of an unknown cost function
    and adjusts parameters to find the optimum. Implements Tustin transform
    filters for better frequency response.
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
        min_output: float = 0.0,
        max_output: float = float('inf')
    ):
        """
        Args:
            sampling_period: Time between updates (seconds)
            dither_frequency: Perturbation frequency (rad/s)
            dither_magnitude: Amplitude of perturbation
            integrator_gain: Gain for parameter updates (negative for minimization)
            initial_value: Starting parameter value
            high_pass_cutoff: High-pass filter cutoff (rad/s)
            low_pass_cutoff: Low-pass filter cutoff (rad/s)
            min_output: Minimum allowed output value
            max_output: Maximum allowed output value
        """
        self.dt = sampling_period
        self.omega = dither_frequency  # Perturbation frequency (rad/s)
        self.a = dither_magnitude  # Perturbation amplitude
        self.k = integrator_gain  # Integrator gain
        self.theta_0 = initial_value  # Initial setpoint
        self.hpf_cutoff = high_pass_cutoff
        self.lpf_cutoff = low_pass_cutoff
        self.min_output = min_output
        self.max_output = max_output

        # Time tracking
        self.time = 0.0
        self.phase = 0.0

        # State variables for Tustin transform filters
        self.prev_cost = 0.0  # Jk-1
        self.prev_hpf_output = 0.0  # sigma_k-1
        self.prev_demodulated = 0.0  # psi_k-1
        self.prev_lpf_output = 0.0  # gamma_k-1
        self.integral = 0.0  # uhat (integrator state)

        # For logging compatibility
        self.m2 = 0.0

    def update(self, cost: float) -> Tuple[float, float, float, float, float, float, float]:
        """
        Update the controller with a new cost measurement.

        Args:
            cost: Current value of the cost function (Jk)

        Returns:
            Tuple containing:
                - output: Perturbed parameter value (setpoint + dither)
                - hpf_output: High-pass filter output (sigma)
                - demodulated: Signal after demodulation (psi)
                - lpf_output: Low-pass filter output / gradient estimate (gamma)
                - amplitude: Probe amplitude
                - raw_gradient: Same as lpf_output
                - setpoint: Current setpoint (without perturbation)
        """
        T = self.dt
        hpf = self.hpf_cutoff
        lpf = self.lpf_cutoff

        # 1. High-pass filter (Tustin/bilinear transform)
        # sigmak = (Jk - Jkm1 - (hpf*T/2 - 1)*sigmakm1) / (1 + hpf*T/2)
        hpf_output = (cost - self.prev_cost - (hpf * T / 2 - 1) * self.prev_hpf_output) / (1 + hpf * T / 2)

        # 2. Demodulate by multiplying with sin(wt)
        demodulated = hpf_output * np.sin(self.omega * self.time)

        # 3. Low-pass filter (Tustin/bilinear transform)
        # gammak = (T*lpf*(psik + psikm1) - (T*lpf - 2)*gammakm1) / (2 + T*lpf)
        lpf_output = (T * lpf * (demodulated + self.prev_demodulated) - (T * lpf - 2) * self.prev_lpf_output) / (2 + T * lpf)

        # 4. Trapezoidal integration
        # uhatk = uhatkm1 + c*T/2*(gammak + gammakm1)
        self.integral = self.integral + self.k * T / 2 * (lpf_output + self.prev_lpf_output)

        # 5. Compute setpoint (base parameter value without perturbation)
        setpoint_raw = self.theta_0 + self.integral

        # 6. Apply output constraints (clamping with anti-windup)
        setpoint = np.clip(setpoint_raw, self.min_output, self.max_output)

        # 7. Anti-windup: correct integrator if output is saturated
        if setpoint_raw < self.min_output:
            self.integral = self.min_output - self.theta_0
        elif setpoint_raw > self.max_output:
            self.integral = self.max_output - self.theta_0

        # 8. Modulation - add perturbation to get final output
        # uk = uhatk + a*sin(wt)
        perturbation = self.a * np.sin(self.omega * self.time)
        output = setpoint + perturbation

        # 9. Update state for next iteration
        self.prev_cost = cost
        self.prev_hpf_output = hpf_output
        self.prev_demodulated = demodulated
        self.prev_lpf_output = lpf_output

        # 10. Update time and phase
        self.time += self.dt
        self.phase = (self.omega * self.time) % (2 * np.pi)
        self.m2 = lpf_output ** 2  # For logging compatibility

        return (
            output,         # Total output (setpoint + perturbation)
            hpf_output,     # High-pass filtered cost (sigma)
            demodulated,    # Demodulated signal (psi)
            lpf_output,     # Gradient estimate (gamma)
            self.a,         # Amplitude
            demodulated,     # Raw gradient (same as lpf_output)
            setpoint        # Base setpoint (no perturbation)
        )

    def reset(self):
        """Reset controller state to initial conditions."""
        self.time = 0.0
        self.phase = 0.0
        self.prev_cost = 0.0
        self.prev_hpf_output = 0.0
        self.prev_demodulated = 0.0
        self.prev_lpf_output = 0.0
        self.integral = 0.0
        self.m2 = 0.0

    def get_state(self) -> dict:
        """Get current controller state."""
        return {
            "time": self.time,
            "phase": self.phase,
            "integral": self.integral,
            "hpf_output": self.prev_hpf_output,
            "lpf_output": self.prev_lpf_output
        }
