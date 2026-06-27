"""
Custom SUMO environment with ambulance prioritization.
Extends sumo-rl SumoEnvironment to inject ambulance state into observations
and apply conditional reward function with potential-based shaping.
"""
import numpy as np
import yaml
from gymnasium import spaces
import sumo_rl
from sumo_rl import SumoEnvironment

from src.environment.observation import AmbulanceObservationFunction
from src.environment.reward import compute_reward


class AmbulancePriorityEnv(SumoEnvironment):
    """
    Single-intersection environment with ambulance priority.

    State: base sumo-rl state + [ambulance_present, ambulance_distance_normalized]
    Reward: conditional on ambulance presence, with potential-based shaping
    """

    def __init__(self, alpha: float, beta: float, gamma: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Dict indexed by ts_id so each traffic signal has its own Phi(s_{t-1}).
        # Using a single float was a bug: with multiple signals (or across reset
        # boundaries) the potential from signal A would bleed into signal B.
        self._prev_potential: dict[str, float] = {}
        # Ambulance state cached from _compute_observations so _compute_rewards
        # doesn't need a second TraCI fanout in the same step.
        self._ambulance_state: dict[str, tuple[bool, float]] = {}

        with open("config/env_config.yaml") as f:
            env_cfg = yaml.safe_load(f)

        self.ambulance_type_id = env_cfg["ambulance_type_id"]

    @property
    def observation_space(self):
        base = super().observation_space
        low = np.append(base.low, [0.0, 0.0])
        high = np.append(base.high, [1.0, 1.0])
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _compute_observations(self):
        observations = {}
        for ts_id, ts in self.traffic_signals.items():
            obs_fn = AmbulanceObservationFunction(ts, self.ambulance_type_id)
            self._ambulance_state[ts_id] = obs_fn._get_ambulance_state()
            observations[ts_id] = obs_fn()
        return observations

    def _compute_rewards(self):
        rewards = {}
        for ts_id, ts in self.traffic_signals.items():
            ambulance_present, ambulance_distance = self._ambulance_state.get(
                ts_id, (False, 0.0)
            )

            reward, new_potential = compute_reward(
                ts=ts,
                ambulance_present=ambulance_present,
                ambulance_distance=ambulance_distance,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                prev_potential=self._prev_potential.get(ts_id, 0.0),
                ambulance_type_id=self.ambulance_type_id,
            )
            self._prev_potential[ts_id] = new_potential
            rewards[ts_id] = reward
        return rewards

    def reset(self, **kwargs):
        self._prev_potential = {}
        self._ambulance_state = {}
        return super().reset(**kwargs)