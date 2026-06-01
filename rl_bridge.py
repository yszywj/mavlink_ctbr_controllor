"""
RL-facing helper layer for CTBRController.

This file intentionally does NOT replace ctbr_controller.py.  It wraps your
existing MAVLink controller and adds the pieces that a reinforcement-learning
environment needs:

1. policy action [-1, 1] -> safe CTBR command mapping
2. observation freshness checks
3. home/goal bookkeeping
4. single-drone safety checks that do not know about the whole multi-agent env
5. recovery-to-home utilities based on local NED position setpoints

Coordinate convention:
- PX4 LOCAL_POSITION_NED is used.
- z is NED, so altitude above home is approximately -z.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Any, Dict

import numpy as np

try:
    from .ctbr_controller import CTBRController
    from .ctbr_tools import ObservationData
except ImportError:  # allows running scripts from repo root during quick tests
    from mavlink_ctbr_controller.ctbr_controller import CTBRController
    from mavlink_ctbr_controller.ctbr_tools import ObservationData


@dataclass
class HomePoint:
    x: float
    y: float
    z: float


@dataclass
class GoalPoint:
    x: float
    y: float
    z: float


@dataclass
class CTBRActionLimits:
    """Safe physical range used after mapping policy actions from [-1, 1]."""

    max_roll_rate: float = 0.35      # rad/s. Start conservative; increase later.
    max_pitch_rate: float = 0.35     # rad/s
    max_yaw_rate: float = 0.25       # rad/s
    hover_thrust: float = 0.56
    thrust_delta: float = 0.08
    thrust_min: float = 0.48
    thrust_max: float = 0.66


@dataclass
class SafetyLimits:
    min_altitude: float = 0.35
    max_altitude: float = 12.0
    max_tilt_deg: float = 70.0
    max_body_rate: float = 7.0
    max_down_speed: float = 4.5
    max_xy_from_home: float = 15.0
    max_z_error_from_home: float = 8.0
    stale_wall_time_sec: float = 0.75


@dataclass
class SafetyResult:
    abnormal: bool
    recoverable: bool
    reason: str = "ok"


@dataclass
class DroneSnapshot:
    obs: ObservationData
    px4_time_boot_ms: int
    wall_time: float
    is_fresh: bool
    armed: bool
    flight_mode: str
    failsafe: bool
    last_status_text: str


@dataclass
class DroneRLState:
    home: Optional[HomePoint] = None
    goal: Optional[GoalPoint] = None
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    last_goal_distance: Optional[float] = None
    last_px4_time_boot_ms: int = 0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def map_policy_action_to_ctbr(
    action: Sequence[float],
    limits: CTBRActionLimits,
) -> Tuple[float, float, float, float]:
    """
    Convert a neural-network action in [-1, 1]^4 into CTBR command values.

    action[0] -> body roll rate
    action[1] -> body pitch rate
    action[2] -> body yaw rate
    action[3] -> normalized thrust around hover_thrust
    """
    if len(action) != 4:
        raise ValueError(f"CTBR policy action must have shape (4,), got {len(action)}")

    a = [clamp(float(v), -1.0, 1.0) for v in action]
    roll_rate = a[0] * limits.max_roll_rate
    pitch_rate = a[1] * limits.max_pitch_rate
    yaw_rate = a[2] * limits.max_yaw_rate
    thrust = limits.hover_thrust + a[3] * limits.thrust_delta
    thrust = clamp(thrust, limits.thrust_min, limits.thrust_max)
    return roll_rate, pitch_rate, yaw_rate, thrust


def observation_vector(
    own: ObservationData,
    other: ObservationData,
    goal: GoalPoint,
    prev_action: np.ndarray,
) -> np.ndarray:
    """
    Low-dimensional actor observation for one drone.

    Layout, length 25:
    own pos(3), own vel(3), own attitude(3), own body rates(3),
    goal relative pos(3), other relative pos(3), other relative vel(3), prev action(4).
    """
    own_pos = np.array([own.x, own.y, own.z], dtype=np.float32)
    own_vel = np.array([own.vx, own.vy, own.vz], dtype=np.float32)
    own_att = np.array([own.roll, own.pitch, own.yaw], dtype=np.float32)
    own_rates = np.array([own.rollspeed, own.pitchspeed, own.yawspeed], dtype=np.float32)

    goal_rel = np.array([goal.x - own.x, goal.y - own.y, goal.z - own.z], dtype=np.float32)
    other_rel_pos = np.array([other.x - own.x, other.y - own.y, other.z - own.z], dtype=np.float32)
    other_rel_vel = np.array([other.vx - own.vx, other.vy - own.vy, other.vz - own.vz], dtype=np.float32)

    return np.concatenate([
        own_pos,
        own_vel,
        own_att,
        own_rates,
        goal_rel,
        other_rel_pos,
        other_rel_vel,
        prev_action.astype(np.float32),
    ]).astype(np.float32)


def goal_distance(obs: ObservationData, goal: GoalPoint) -> float:
    return math.sqrt((obs.x - goal.x) ** 2 + (obs.y - goal.y) ** 2 + (obs.z - goal.z) ** 2)


def inter_drone_distance(obs_a: ObservationData, obs_b: ObservationData) -> float:
    return math.sqrt((obs_a.x - obs_b.x) ** 2 + (obs_a.y - obs_b.y) ** 2 + (obs_a.z - obs_b.z) ** 2)


class CTBRDroneRLAdapter:
    """Small wrapper around one CTBRController for RL use."""

    def __init__(
        self,
        drone_id: int,
        controller: CTBRController,
        action_limits: Optional[CTBRActionLimits] = None,
        safety_limits: Optional[SafetyLimits] = None,
    ):
        self.drone_id = int(drone_id)
        self.controller = controller
        self.action_limits = action_limits or CTBRActionLimits()
        self.safety_limits = safety_limits or SafetyLimits()
        self.state = DroneRLState()

    def start_io(self, data_stream_hz: int = 30, start_logging: bool = True) -> None:
        if not self.controller.is_monitoring:
            self.controller.start_monitoring(freq_hz=data_stream_hz)
        if start_logging:
            self.controller.start_logging()

    def stop_ctbr(self) -> None:
        if self.controller.is_sending:
            self.controller.stop_ctbr_send_thread()

    def start_ctbr(self, hz: int) -> None:
        if not self.controller.is_sending:
            self.controller.start_ctbr_send_thread(frequency=hz)

    def set_safe_ctbr(self) -> None:
        self.controller.update_ctbr_send_params(
            body_roll_rate=0.0,
            body_pitch_rate=0.0,
            body_yaw_rate=0.0,
            thrust=self.action_limits.hover_thrust,
        )
        self.state.prev_action = np.zeros(4, dtype=np.float32)

    def apply_policy_action(self, action: Sequence[float]) -> Tuple[float, float, float, float]:
        roll, pitch, yaw, thrust = map_policy_action_to_ctbr(action, self.action_limits)
        self.controller.update_ctbr_send_params(
            body_roll_rate=roll,
            body_pitch_rate=pitch,
            body_yaw_rate=yaw,
            thrust=thrust,
        )
        self.state.prev_action = np.array([roll, pitch, yaw, thrust], dtype=np.float32)
        return roll, pitch, yaw, thrust

    def get_observation(self) -> ObservationData:
        if not self.controller.data_sync:
            raise RuntimeError(f"drone {self.drone_id}: data_sync is not enabled")
        return self.controller.data_sync.get_latest_observation()

    def snapshot(self) -> DroneSnapshot:
        obs = self.get_observation()
        now_wall = time.time()
        last_update = 0.0
        px4_time = 0
        if self.controller.data_sync:
            last_update = getattr(self.controller.data_sync, "_last_update_wall_time", 0.0)
            px4_time = self.controller.data_sync.get_latest_px4_time_ms()

        is_fresh = (now_wall - last_update) <= self.safety_limits.stale_wall_time_sec
        armed = bool(getattr(self.controller, "_armed", False))
        flight_mode = self.controller._flight_mode_name() if hasattr(self.controller, "_flight_mode_name") else ""
        recent_text = " | ".join(self.controller._recent_status_text(10)) if hasattr(self.controller, "_recent_status_text") else ""
        recent_lower = recent_text.lower()
        failsafe = any(k in recent_lower for k in ["failsafe", "rtl", "battery warning", "land"])
        return DroneSnapshot(
            obs=obs,
            px4_time_boot_ms=px4_time,
            wall_time=now_wall,
            is_fresh=is_fresh,
            armed=armed,
            flight_mode=flight_mode,
            failsafe=failsafe,
            last_status_text=recent_text,
        )

    def capture_home(self) -> HomePoint:
        obs = self.get_observation()
        self.state.home = HomePoint(float(obs.x), float(obs.y), float(obs.z))
        return self.state.home

    def set_goal(self, goal: GoalPoint) -> None:
        self.state.goal = goal
        obs = self.get_observation()
        self.state.last_goal_distance = goal_distance(obs, goal)

    def check_single_drone_safety(self) -> SafetyResult:
        snap = self.snapshot()
        obs = snap.obs
        limits = self.safety_limits

        if not snap.is_fresh:
            return SafetyResult(True, False, "stale_observation")
        if not snap.armed:
            return SafetyResult(True, False, "disarmed")
        if snap.failsafe:
            return SafetyResult(True, False, f"px4_failsafe_status: {snap.last_status_text}")

        alt = -float(obs.z)
        tilt_deg = math.degrees(math.sqrt(float(obs.roll) ** 2 + float(obs.pitch) ** 2))
        max_rate = max(abs(float(obs.rollspeed)), abs(float(obs.pitchspeed)), abs(float(obs.yawspeed)))

        if alt < limits.min_altitude:
            return SafetyResult(True, False, f"near_ground_alt={alt:.2f}m")
        if alt > limits.max_altitude:
            return SafetyResult(True, True, f"too_high_alt={alt:.2f}m")
        if tilt_deg > limits.max_tilt_deg:
            return SafetyResult(True, True, f"tilt_too_large={tilt_deg:.1f}deg")
        if max_rate > limits.max_body_rate:
            return SafetyResult(True, True, f"body_rate_too_large={max_rate:.2f}rad/s")
        if float(obs.vz) > limits.max_down_speed:  # NED vz > 0 means downward
            return SafetyResult(True, True, f"falling_fast_vz={obs.vz:.2f}m/s")

        if self.state.home is not None:
            home = self.state.home
            xy_dist = math.sqrt((float(obs.x) - home.x) ** 2 + (float(obs.y) - home.y) ** 2)
            z_err = abs(float(obs.z) - home.z)
            if xy_dist > limits.max_xy_from_home:
                return SafetyResult(True, True, f"xy_out_of_bounds={xy_dist:.2f}m")
            if z_err > limits.max_z_error_from_home:
                return SafetyResult(True, True, f"z_out_of_bounds={z_err:.2f}m")

        return SafetyResult(False, True, "ok")

    def recover_to_home(
        self,
        time_keeper: Any,
        timeout_sim_sec: float = 10.0,
        tolerance_m: float = 0.5,
    ) -> bool:
        if self.state.home is None:
            raise RuntimeError(f"drone {self.drone_id}: home is not set")

        home = self.state.home
        self.stop_ctbr()
        self.set_safe_ctbr()
        if hasattr(self.controller, "set_episode_phase"):
            self.controller.set_episode_phase("recover")

        ok_mode = self.controller.change_control_mode(
            mode=6,
            is_maintain_offboard=False,  # avoid asyncio task in non-async MAPPO runners
            default_x=home.x,
            default_y=home.y,
            default_z=home.z,
        )
        if not ok_mode:
            return False

        start_ms = time_keeper.now_ms()
        timeout_ms = int(timeout_sim_sec * 1000)
        while time_keeper.now_ms() - start_ms < timeout_ms:
            safety = self.check_single_drone_safety()
            if safety.abnormal and not safety.recoverable:
                return False

            self.controller.send_hover_setpoint(home.x, home.y, home.z)
            obs = self.get_observation()
            err = math.sqrt((obs.x - home.x) ** 2 + (obs.y - home.y) ** 2 + (obs.z - home.z) ** 2)
            if err < tolerance_m:
                return True
            time_keeper.wait(0.1, timeout=2.0)

        return False

    def cleanup(self) -> None:
        self.stop_ctbr()
        self.controller.cleanup()
