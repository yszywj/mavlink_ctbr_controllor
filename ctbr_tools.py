from dataclasses import dataclass


@dataclass
class ControlParams:
    body_roll_rate: float = 0.0
    body_pitch_rate: float = 0.0
    body_yaw_rate: float = 0.0
    thrust: float = 0.55