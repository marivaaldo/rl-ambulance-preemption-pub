"""
Custom observation space extending sumo-rl's default.
Adds ambulance presence (binary) and relative position (normalized distance).
"""
import numpy as np
from sumo_rl.environment.observations import DefaultObservationFunction
from gymnasium import spaces
import traci


class AmbulanceObservationFunction(DefaultObservationFunction):

    def __init__(self, ts, ambulance_type_id: str):
        super().__init__(ts)
        self.ambulance_type_id = ambulance_type_id

    def __call__(self) -> np.ndarray:
        base_obs = super().__call__()
        ambulance_present, normalized_distance = self._get_ambulance_state()
        ambulance_features = np.array(
            [float(ambulance_present), normalized_distance], dtype=np.float32
        )
        return np.concatenate([base_obs, ambulance_features])

    def observation_space(self) -> spaces.Box:
        base_space = super().observation_space()
        low = np.append(base_space.low, [0.0, 0.0])
        high = np.append(base_space.high, [1.0, 1.0])
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_ambulance_state(self) -> tuple[bool, float]:
        """Returns (present, normalized_distance) for any ambulance in controlled lanes."""
        for lane_id in self.ts.lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            for veh_id in vehicle_ids:
                try:
                    veh_type = traci.vehicle.getTypeID(veh_id)
                    if veh_type == self.ambulance_type_id:
                        distance = traci.vehicle.getLanePosition(veh_id)
                        lane_length = traci.lane.getLength(lane_id)
                        return True, min(distance / lane_length, 1.0)
                except traci.TraCIException:
                    continue
        return False, 0.0