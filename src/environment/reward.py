"""
Conditional reward function with ambulance prioritization.
Implements potential-based reward shaping (Ng et al., 1999) for sparse emergency events.
"""
import traci


def compute_reward(
    ts,
    ambulance_present: bool,
    ambulance_distance: float,
    alpha: float,
    beta: float,
    gamma: float,
    prev_potential: float,
    ambulance_type_id: str,
) -> tuple[float, float]:
    """
    Computes conditional reward and returns (reward, new_potential).

    When ambulance is present:
        r = -alpha * T_amb - beta * D_traf
    When absent:
        r = -beta * D_traf

    Shaped reward: r' = r + gamma * Phi(s') - Phi(s)
    where Phi(s) = ambulance_distance if ambulance_present else 0.0

    Phi is defined only over observable features (ambulance_present, ambulance_distance),
    satisfying Ng et al. (1999): no spurious inter-episode credit because ambulance_present
    is part of the state and Phi=0 when absent.
    Positive sign: as ambulance approaches (distance → 1), Phi grows → shaping is positive
    → agent is rewarded for actions that allow the ambulance to advance.
    """
    avg_delay = _compute_average_delay(ts)
    ambulance_waiting = _compute_ambulance_waiting(ts, ambulance_type_id)

    if ambulance_present:
        base_reward = -alpha * ambulance_waiting - beta * avg_delay
    else:
        base_reward = -beta * avg_delay

    new_potential = ambulance_distance if ambulance_present else 0.0

    # When the ambulance exits (prev_potential > 0 but ambulance_present = False),
    # the standard γ·Φ(s') − Φ(s) produces a spurious negative pulse that punishes
    # a success event. Zero out shaping on this single transition.
    ambulance_just_exited = (not ambulance_present) and (prev_potential > 0.0)
    shaping = 0.0 if ambulance_just_exited else (gamma * new_potential - prev_potential)

    shaped_reward = base_reward + shaping

    return shaped_reward, new_potential


def _compute_average_delay(ts) -> float:
    """Average waiting time across all vehicles in controlled lanes."""
    total_waiting = 0.0
    total_vehicles = 0
    for lane_id in ts.lanes:
        vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
        for veh_id in vehicles:
            try:
                total_waiting += traci.vehicle.getWaitingTime(veh_id)
                total_vehicles += 1
            except traci.TraCIException:
                continue
    return total_waiting / max(total_vehicles, 1)


def _compute_ambulance_waiting(ts, ambulance_type_id: str) -> float:
    """Returns accumulated waiting time of the ambulance in controlled lanes.

    Uses getAccumulatedWaitingTime (total since network entry) rather than
    getWaitingTime (current continuous stop), to penalise total episode delay.
    """
    for lane_id in ts.lanes:
        vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
        for veh_id in vehicles:
            try:
                veh_type = traci.vehicle.getTypeID(veh_id)
                if veh_type == ambulance_type_id:
                    return traci.vehicle.getAccumulatedWaitingTime(veh_id)
            except traci.TraCIException:
                continue
    return 0.0