"""
Quake Quantistic 3D Time (QQ3DT)
Experiment 2: Predictive Temporal Feedback

This module implements a dynamical system that integrates
an internal probabilistic projection of its future state
as a feedback mechanism.

NOTE:
- This is a computational simulation.
- No physical claim about time dimensionality is made.
"""

import numpy as np


class TemporalFeedbackSystem:
    def __init__(
        self,
        initial_state,
        system_model,
        predictor,
        alpha=0.3,
        noise_std=0.0
    ):
        """
        Parameters
        ----------
        initial_state : array-like
            Initial state of the system.
        system_model : callable
            Natural system dynamics: f(state, dt) -> new_state
        predictor : callable
            Internal future projection model: g(state, dt) -> predicted_state
        alpha : float
            Feedback strength (0 = no feedback).
        noise_std : float
            Standard deviation of Gaussian noise added to dynamics.
        """
        self.state = np.array(initial_state, dtype=float)
        self.model = system_model
        self.predictor = predictor
        self.alpha = alpha
        self.noise_std = noise_std

        self.history = [self.state.copy()]
        self.prediction_history = []
        self.stability_history = []

    def step(self, dt=1.0):
        """
        Perform one time step of evolution.
        """
        # Natural evolution
        natural_state = self.model(self.state, dt)

        # Add noise (environmental perturbation)
        if self.noise_std > 0:
            natural_state = natural_state + np.random.normal(
                0, self.noise_std, size=self.state.shape
            )

        # Internal predictive projection (not actual future)
        predicted_future = self.predictor(self.state, dt)

        # Temporal feedback integration
        new_state = (
            (1.0 - self.alpha) * natural_state
            + self.alpha * predicted_future
        )

        # Stability metric
        stability = self._compute_stability(self.state, new_state)

        # Update internal state
        self.state = new_state

        # Logging
        self.history.append(new_state.copy())
        self.prediction_history.append(predicted_future.copy())
        self.stability_history.append(stability)

        return new_state, stability

    def run(self, steps=100, dt=1.0):
        """
        Run the system for multiple steps.
        """
        for _ in range(steps):
            self.step(dt)

        return {
            "history": np.array(self.history),
            "predictions": np.array(self.prediction_history),
            "stability": np.array(self.stability_history),
        }

    @staticmethod
    def _compute_stability(prev_state, new_state):
        """
        Simple stability metric:
        inverse of state displacement magnitude.
        """
        delta = np.linalg.norm(new_state - prev_state)
        return 1.0 / (delta + 1e-8)


# --------------------------------------------------
# Example system: damped harmonic oscillator
# --------------------------------------------------

def harmonic_oscillator(state, dt):
    """
    State = [position, velocity]
    """
    x, v = state
    k = 1.0      # spring constant
    gamma = 0.15 # damping

    dx = v
    dv = -k * x - gamma * v

    return state + dt * np.array([dx, dv])


def noisy_predictor(state, dt):
    """
    Imperfect internal model of the future.
    """
    prediction = harmonic_oscillator(state, dt)
    noise = np.random.normal(0, 0.05, size=state.shape)
    return prediction + noise


# --------------------------------------------------
# Standalone test
# --------------------------------------------------

if __name__ == "__main__":
    system = TemporalFeedbackSystem(
        initial_state=[1.0, 0.0],
        system_model=harmonic_oscillator,
        predictor=noisy_predictor,
        alpha=0.4,
        noise_std=0.02
    )

    results = system.run(steps=200, dt=0.05)

    print("Final state:", results["history"][-1])
    print("Mean stability:", np.mean(results["stability"]))
