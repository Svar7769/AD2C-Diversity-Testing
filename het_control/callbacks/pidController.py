"""
PID Controller for automatic tuning of desired SND parameter.
"""
import numpy as np


class PIDController:
    """
    Simple PID controller for parameter optimization.
    
    The controller minimizes a cost function by adjusting a parameter
    using proportional, integral, and derivative control terms.
    """
    
    def __init__(
        self,
        kp: float = 0.01,
        ki: float = 0.001,
        kd: float = 0.0,
        initial_value: float = 1.0,
        target_reward: float = 3.2,
        min_output: float = 0.0,
        max_output: float = 3.0,
        integral_clamp: float = 10.0,
        derivative_filter_alpha: float = 0.1
    ):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            initial_value: Starting parameter value
            target_reward: Target/maximum reward to achieve (setpoint for error calculation)
            min_output: Minimum output value
            max_output: Maximum output value
            integral_clamp: Maximum absolute value for integral term
            derivative_filter_alpha: Smoothing factor for derivative (0=no filter, 1=no change)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_reward = target_reward
        self.min_output = min_output
        self.max_output = max_output
        self.integral_clamp = integral_clamp
        self.derivative_filter_alpha = derivative_filter_alpha
        
        # State variables
        self.setpoint = initial_value
        self.integral = 0.0
        self.previous_error = 0.0
        self.filtered_derivative = 0.0
        
        # For tracking
        self.last_reward = 0.0
        self.iteration = 0
    
    def update(self, current_reward: float) -> tuple[float, float, float, float, float]:
        """
        Update controller with new reward measurement.
        
        The PID controller adjusts the SND parameter to drive the reward towards
        the target reward. Error = target_reward - current_reward.
        
        Args:
            current_reward: Current mean reward achieved
            
        Returns:
            Tuple of (output, error, p_term, i_term, d_term)
            - output: New SND parameter value (clamped to bounds)
            - error: Error signal (target_reward - current_reward)
            - p_term: Proportional term contribution
            - i_term: Integral term contribution
            - d_term: Derivative term contribution
        """
        # Error is the difference between target and actual reward
        # Positive error means we're below target (need to adjust SND)
        # Negative error means we're above target
        error = - (self.target_reward - current_reward)
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup clamping
        self.integral += error
        self.integral = np.clip(self.integral, -self.integral_clamp, self.integral_clamp)
        i_term = self.ki * self.integral
        
        # Derivative term with filtering to reduce noise
        derivative = error - self.previous_error
        self.filtered_derivative = (
            self.derivative_filter_alpha * self.filtered_derivative +
            (1 - self.derivative_filter_alpha) * derivative
        )
        d_term = self.kd * self.filtered_derivative
        
        # Compute control signal
        # The control signal adjusts SND based on the error
        # Note: The relationship between SND and reward is task-dependent
        # If higher SND -> higher reward: control signal should be positive when error > 0
        # If higher SND -> lower reward: you may need to negate the control signal
        control_signal = p_term + i_term + d_term
        
        # Update setpoint (SND parameter)
        self.setpoint += control_signal
        
        # Clamp output to bounds
        output = np.clip(self.setpoint, self.min_output, self.max_output)
        
        # Update state for next iteration
        self.previous_error = error
        self.last_reward = current_reward
        self.iteration += 1
        
        return output, error, p_term, i_term, d_term
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.filtered_derivative = 0.0
        self.iteration = 0