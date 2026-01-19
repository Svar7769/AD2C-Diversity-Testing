"""
Extremum Seeking Control (ESC) with ML-style optimizers.
Modular implementation with separate optimizer classes.
"""
import numpy as np
from typing import Tuple, Literal


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


class RMSpropOptimizer:
    """RMSprop optimizer for ESC parameter updates."""
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
        centered: bool = False
    ):
        self.lr = learning_rate
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.centered = centered
        
        # State
        self.v = 0.0
        self.buf = 0.0
        self.g_avg = 0.0
        
    def step(self, gradient: float, parameter: float) -> float:
        # Update moving average of squared gradients
        self.v = self.alpha * self.v + (1.0 - self.alpha) * (gradient ** 2)
        
        # Centered variant
        if self.centered:
            self.g_avg = self.alpha * self.g_avg + (1.0 - self.alpha) * gradient
            v_centered = self.v - (self.g_avg ** 2)
            denominator = np.sqrt(v_centered + self.eps)
        else:
            denominator = np.sqrt(self.v + self.eps)
        
        # Apply update
        if self.momentum > 0:
            self.buf = self.momentum * self.buf + gradient / denominator
            update = self.lr * self.buf
        else:
            update = self.lr * gradient / denominator
        
        return parameter - update
    
    def reset(self):
        self.v = 0.0
        self.buf = 0.0
        self.g_avg = 0.0
    
    def get_state(self) -> dict:
        return {
            "v": self.v,
            "momentum_buffer": self.buf,
            "gradient_avg": self.g_avg if self.centered else None,
            "learning_rate": self.lr
        }
    
    def set_learning_rate(self, lr: float):
        self.lr = lr


class AdamOptimizer:
    """Adam optimizer for ESC parameter updates."""
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad: bool = False
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad = amsgrad
        
        # State
        self.m = 0.0
        self.v = 0.0
        self.v_max = 0.0
        self.t = 0
        
    def step(self, gradient: float, parameter: float) -> float:
        self.t += 1
        
        # Update moments
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (gradient ** 2)
        
        # Bias correction
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        
        if self.amsgrad:
            self.v_max = max(self.v_max, self.v)
            v_hat = self.v_max / (1.0 - self.beta2 ** self.t)
        else:
            v_hat = self.v / (1.0 - self.beta2 ** self.t)
        
        # Compute update
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return parameter - update
    
    def reset(self):
        self.m = 0.0
        self.v = 0.0
        self.v_max = 0.0
        self.t = 0
    
    def get_state(self) -> dict:
        return {
            "momentum": self.m,
            "variance": self.v,
            "v_max": self.v_max if self.amsgrad else None,
            "t": self.t,
            "learning_rate": self.lr
        }
    
    def set_learning_rate(self, lr: float):
        self.lr = lr


class SGDOptimizer:
    """SGD optimizer with momentum for ESC parameter updates."""
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False
    ):
        self.lr = learning_rate
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        
        # State
        self.buf = 0.0
        
    def step(self, gradient: float, parameter: float) -> float:
        if self.momentum > 0:
            if self.buf == 0.0:
                self.buf = gradient
            else:
                self.buf = self.momentum * self.buf + (1.0 - self.dampening) * gradient
            
            if self.nesterov:
                update = self.lr * (gradient + self.momentum * self.buf)
            else:
                update = self.lr * self.buf
        else:
            update = self.lr * gradient
        
        return parameter - update
    
    def reset(self):
        self.buf = 0.0
    
    def get_state(self) -> dict:
        return {
            "momentum_buffer": self.buf,
            "learning_rate": self.lr
        }
    
    def set_learning_rate(self, lr: float):
        self.lr = lr


class ExtremumSeekingController:
    """
    Extremum Seeking Controller with ML-style optimizers.
    
    Uses sinusoidal perturbations to estimate gradients and
    optimizers (Adam, RMSprop, SGD) to update parameters.
    """
    
    def __init__(
        self,
        sampling_period: float,
        dither_frequency: float,
        dither_magnitude: float,
        initial_value: float,
        high_pass_cutoff: float,
        low_pass_cutoff: float,
        min_output: float = 0.0,
        maximize: bool = False,
        optimizer_type: Literal['adam', 'rmsprop', 'sgd'] = 'adam',
        learning_rate: float = 0.01,
        **optimizer_kwargs
    ):
        """
        Args:
            sampling_period: Time between updates (seconds)
            dither_frequency: Perturbation frequency (rad/s)
            dither_magnitude: Amplitude of perturbation
            initial_value: Starting parameter value
            high_pass_cutoff: High-pass filter cutoff (rad/s)
            low_pass_cutoff: Low-pass filter cutoff (rad/s)
            min_output: Minimum allowed output value
            maximize: If True, maximize objective; if False, minimize
            optimizer_type: Type of optimizer ('adam', 'rmsprop', 'sgd')
            learning_rate: Learning rate for optimizer
            **optimizer_kwargs: Additional optimizer-specific arguments
        """
        self.dt = sampling_period
        self.omega = dither_frequency
        self.a = dither_magnitude
        self.min_output = min_output
        self.maximize = maximize
        self.optimizer_type = optimizer_type
        
        # Initialize filters
        self.hpf = HighPassFilter(sampling_period, high_pass_cutoff)
        self.lpf = LowPassFilter(sampling_period, low_pass_cutoff)
        
        # State variables
        self.theta = initial_value
        self.phase = 0.0
        
        # Gradient tracking
        self.m2 = 0.0
        self.beta = 0.8
        self.epsilon = 1e-8
        
        # Initialize optimizer
        if optimizer_type == 'rmsprop':
            self.optimizer = RMSpropOptimizer(learning_rate=learning_rate, **optimizer_kwargs)
        elif optimizer_type == 'adam':
            self.optimizer = AdamOptimizer(learning_rate=learning_rate, **optimizer_kwargs)
        elif optimizer_type == 'sgd':
            self.optimizer = SGDOptimizer(learning_rate=learning_rate, **optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        self.step_count = 0
    
    def update(self, cost: float) -> Tuple[float, float, float, float, float, float]:
        """
        Update controller with new cost/reward measurement.
        
        Args:
            cost: Current value of objective function
            
        Returns:
            Tuple of (output, hpf_output, gradient_estimate, gradient_magnitude, 
                     raw_gradient, theta_value)
        """
        # ESC gradient estimation
        hpf_output = self.hpf.apply(cost)
        demodulated = hpf_output * np.sin(self.phase)
        gradient_estimate = self.lpf.apply(demodulated)
        
        # Track gradient magnitude
        self.m2 = self.beta * self.m2 + (1.0 - self.beta) * (gradient_estimate ** 2)
        gradient_magnitude = np.sqrt(self.m2 + self.epsilon)
        
        # Apply optimizer
        if self.maximize:
            gradient = -gradient_estimate  # Negate for gradient ascent
        else:
            gradient = gradient_estimate  # Normal gradient descent
        
        self.theta = self.optimizer.step(gradient, self.theta)
        
        # Apply constraints
        self.theta = max(self.theta, self.min_output)
        
        # Add perturbation
        perturbation = self.a * np.sin(self.phase)
        output = self.theta + perturbation
        
        # Update phase
        self.phase += self.omega * self.dt
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi
        
        self.step_count += 1
        
        return (
            output,
            hpf_output,
            gradient_estimate,
            gradient_magnitude,
            gradient_estimate,
            self.theta
        )
    
    def reset(self):
        """Reset controller state."""
        self.hpf.reset()
        self.lpf.reset()
        self.phase = 0.0
        self.m2 = 0.0
        self.theta = 0.0
        self.optimizer.reset()
        self.step_count = 0
    
    def get_state(self) -> dict:
        """Get current controller state."""
        state = {
            "phase": self.phase,
            "theta": self.theta,
            "gradient_magnitude": np.sqrt(self.m2 + self.epsilon),
            "optimizer_type": self.optimizer_type,
            "step_count": self.step_count
        }
        
        # Add optimizer state
        optimizer_state = self.optimizer.get_state()
        state.update(optimizer_state)
        
        return state
    
    def set_learning_rate(self, lr: float):
        """Update optimizer learning rate."""
        self.optimizer.set_learning_rate(lr)