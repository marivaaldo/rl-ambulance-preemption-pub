"""
Traffic metrics computation from SUMO simulation.
Collected at each step and aggregated over evaluation episodes.
"""
from dataclasses import dataclass, field
from typing import Optional
import traci
import numpy as np


@dataclass
class EpisodeMetrics:
    num_seconds: float = 3600.0

    ambulance_transit_times: list = field(default_factory=list)
    queue_lengths: list = field(default_factory=list)

    # Counts stopped→moving transitions, not per-step stopped state.
    ambulance_stop_transitions: int = 0
    _ambulance_was_stopped: bool = field(default=False, repr=False)

    vehicle_throughput: int = 0

    _ambulance_entry_time: Optional[float] = field(default=None, repr=False)
    _ambulance_id: Optional[str] = field(default=None, repr=False)

    _accumulated_wait: dict = field(default_factory=dict, repr=False)
    _trip_delays_completed: list = field(default_factory=list, repr=False)
    _trip_delays_truncated: list = field(default_factory=list, repr=False)

    _action_runs: dict = field(default_factory=dict, repr=False)
    _current_action: Optional[int] = field(default=None, repr=False)
    _current_action_steps: int = field(default=0, repr=False)

    def record_action(self, action: int) -> None:
        """Track consecutive action runs. Call before update_step each step."""
        if self._current_action is None:
            self._current_action = action
            self._current_action_steps = 1
        elif action == self._current_action:
            self._current_action_steps += 1
        else:
            # Action changed: store the completed run
            self._action_runs.setdefault(self._current_action, []).append(
                self._current_action_steps
            )
            self._current_action = action
            self._current_action_steps = 1

    def update_step(self, sim_time: float, ambulance_type_id: str) -> None:
        """Update metrics at each simulation step."""

        all_vehicles = traci.vehicle.getIDList()

        if self._ambulance_id is not None and self._ambulance_id not in all_vehicles:
            self.record_ambulance_exit(sim_time)

        departed = set(self._accumulated_wait.keys()) - set(all_vehicles)
        for veh_id in departed:
            self._trip_delays_completed.append(self._accumulated_wait.pop(veh_id))

        for veh_id in all_vehicles:
            try:
                # getAccumulatedWaitingTime (total since entry) rather than
                # getWaitingTime (resets on movement) for correct per-trip delay.
                self._accumulated_wait[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)

                veh_type = traci.vehicle.getTypeID(veh_id)
                if veh_type == ambulance_type_id:
                    self._ambulance_id = veh_id
                    if self._ambulance_entry_time is None:
                        self._ambulance_entry_time = sim_time
                    is_stopped = traci.vehicle.getSpeed(veh_id) < 0.1
                    if is_stopped and not self._ambulance_was_stopped:
                        self.ambulance_stop_transitions += 1
                    self._ambulance_was_stopped = is_stopped

            except traci.TraCIException:
                self._accumulated_wait.pop(veh_id, None)
                continue

        stopped = sum(1 for v in all_vehicles if traci.vehicle.getSpeed(v) < 0.1)
        self.queue_lengths.append(stopped)
        self.vehicle_throughput += traci.simulation.getArrivedNumber()

    def finalize_vehicle_delays(self) -> None:
        """Register delays of vehicles still in network at episode end. Call before summarize()."""
        for veh_id, acc_wait in self._accumulated_wait.items():
            self._trip_delays_truncated.append(acc_wait)
        self._accumulated_wait.clear()

    def record_ambulance_exit(self, sim_time: float) -> None:
        """Call when ambulance leaves the intersection area."""
        if self._ambulance_entry_time is not None:
            transit = sim_time - self._ambulance_entry_time
            self.ambulance_transit_times.append(transit)
            self._ambulance_entry_time = None
            self._ambulance_id = None
            self._ambulance_was_stopped = False

    def summarize(self) -> dict:
        all_delays = self._trip_delays_completed + self._trip_delays_truncated
        throughput_veh_h = self.vehicle_throughput / self.num_seconds * 3600.0

        # Flush any open action run before summarizing.
        action_runs = {k: list(v) for k, v in self._action_runs.items()}
        if self._current_action is not None and self._current_action_steps > 0:
            action_runs.setdefault(self._current_action, []).append(self._current_action_steps)

        phase_metrics = {}
        action_labels = {0: "ns", 1: "ew"}
        delta_time = 5  # matches env_config.yaml delta_time
        for action_idx, runs in action_runs.items():
            label = action_labels.get(action_idx, str(action_idx))
            durations_s = [r * delta_time for r in runs]
            phase_metrics[f"action_{label}_mean_run_s"] = float(np.mean(durations_s))
            phase_metrics[f"action_{label}_std_run_s"] = float(np.std(durations_s))
            phase_metrics[f"action_{label}_n_runs"] = len(runs)
            phase_metrics[f"action_{label}_total_s"] = float(np.sum(durations_s))
            phase_metrics[f"action_{label}_pct"] = float(
                np.sum(durations_s) / self.num_seconds * 100
            )

        return {
            "mean_ambulance_transit_s": float(np.mean(self.ambulance_transit_times)) if self.ambulance_transit_times else None,
            "std_ambulance_transit_s": float(np.std(self.ambulance_transit_times)) if self.ambulance_transit_times else None,
            "ambulance_stop_transitions": self.ambulance_stop_transitions,
            "mean_vehicle_delay_completed_s": float(np.mean(self._trip_delays_completed)) if self._trip_delays_completed else 0.0,
            "std_vehicle_delay_completed_s": float(np.std(self._trip_delays_completed)) if self._trip_delays_completed else 0.0,
            "mean_vehicle_delay_all_s": float(np.mean(all_delays)) if all_delays else 0.0,
            "std_vehicle_delay_all_s": float(np.std(all_delays)) if all_delays else 0.0,
            "mean_queue_length": float(np.mean(self.queue_lengths)) if self.queue_lengths else 0.0,
            "vehicle_throughput": self.vehicle_throughput,
            "vehicle_throughput_veh_h": float(throughput_veh_h),
            **phase_metrics,
        }
