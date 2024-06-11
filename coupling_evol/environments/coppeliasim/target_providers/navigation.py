import numpy as np

from coupling_evol.engine.embedded_control import EmbeddedTargetParameter
from coupling_evol.engine.experiment_executor import TargetProvider
from coupling_evol.environments.coppeliasim.coppeliasim_environment import Y_DIMENSION, XYZ_DIMENSION, Q_DIMENSION, get_heading
import coupling_evol.engine.common as C

import coupling_evol.data_process.inprocess.recorder as rlog
from coupling_evol.engine import common as C
REC = rlog.get_recorder(C.RecordNamespace.TARGET_PROVIDER.key)

def position_and_heading(data):
    position = data[Y_DIMENSION:Y_DIMENSION+XYZ_DIMENSION]
    q = data[Y_DIMENSION + XYZ_DIMENSION:Y_DIMENSION + XYZ_DIMENSION + Q_DIMENSION]
    heading = get_heading(q)
    return position, heading

R_FORWARD_VEL_REF = "vel_ref_fwd"
R_SIDE_VEL_REF = "vel_ref_side"
R_TURN_VELOCITY_REF = "vel_ref_trn"
R_POS_XYZ = "pos_xyz"
R_POS_RPY = "pos_rpy"
R_GOAL_XY = "goal_xy"

class TurnAndGo(TargetProvider):
    def __init__(self, sensory_dim, granularity, xy_goal=(0, 0), max_linear_velocity=2., max_turn_velocity=1.,
                 zero_collapse_epsilon=0.01, vel_w=1., ang_z_w=1., ang_w=0.1, eff_w=0.1, vel_side_w=None):
        super().__init__()
        self.sensory_dim = sensory_dim
        self.granularity = granularity
        self.xy_goal = np.asarray(xy_goal)
        self.max_linear_velocity = max_linear_velocity
        self.max_turn_velocity = max_turn_velocity
        self.zero_collapse_epsilon = zero_collapse_epsilon
        self.vel_w = vel_w
        if vel_side_w is not None:
            self.vel_side_w = vel_side_w
        else:
            self.vel_side_w = vel_w
        self.ang_z_w = ang_z_w
        self.ang_w = ang_w
        self.eff_w = eff_w

    def navigate(self, xyz, roll_pitch_yaw):
        dlt_loc = self.xy_goal - xyz[0:2]
        trg_head = np.arctan2(dlt_loc[1], dlt_loc[0])
        # dlt_head = trg_head - roll_pitch_yaw[2]
        _dlt_head = trg_head - roll_pitch_yaw[2]
        dlt_head = np.arctan2(np.sin(_dlt_head), np.cos(_dlt_head))

        forward_vel = np.maximum(np.linalg.norm(dlt_loc) * np.cos(dlt_head), 0)
        turn_vel = dlt_head

        # norming
        forward_vel = np.tanh(forward_vel) * self.max_linear_velocity
        turn_vel = np.tanh(turn_vel) * self.max_turn_velocity

        # zero collapse
        if np.abs(forward_vel) < self.zero_collapse_epsilon:
            forward_vel = 0.
        if np.abs(turn_vel) < self.zero_collapse_epsilon:
            turn_vel = 0.
        return forward_vel, 0., turn_vel

    def __call__(self, data: np.ndarray) -> EmbeddedTargetParameter:
        xyz, rpy = position_and_heading(data)
        forward_vel, side_vel, turn_vel = self.navigate(xyz, rpy)
        REC(R_POS_RPY, rpy)
        REC(R_POS_XYZ, xyz)
        REC(R_GOAL_XY, self.xy_goal)
        REC(R_FORWARD_VEL_REF, forward_vel)
        REC(R_SIDE_VEL_REF, side_vel)
        REC(R_TURN_VELOCITY_REF, turn_vel)
        return motion_target(
            sensory_dimension=self.sensory_dim,
            granularity=self.granularity,
            vel_head_val=forward_vel,
            vel_side_val=side_vel,
            vel_direrction_val=turn_vel,
            vel_w=self.vel_w, ang_z_w=self.ang_z_w, ang_w=self.ang_w, eff_w=self.eff_w,
            vel_side_w=self.vel_side_w, 
            vel_direction_metric_bounded=False,
            vel_head_metric_bounded=False
        )


class BoundedTurnAndGo(TurnAndGo):
    def __init__(self, sensory_dim, granularity, xy_goal=(0, 0), max_linear_velocity=2., max_turn_velocity=1.,
                 zero_collapse_epsilon=0.01, vel_w=1., ang_z_w=1., ang_w=0.1, eff_w=0.1, vel_side_w=None):
        super().__init__(sensory_dim, granularity, xy_goal=xy_goal, max_linear_velocity=max_linear_velocity,
                         max_turn_velocity=max_turn_velocity, zero_collapse_epsilon=zero_collapse_epsilon,
                         vel_w=vel_w, ang_z_w=ang_z_w, ang_w=ang_w, eff_w=eff_w, vel_side_w=vel_side_w)
        self.prev_dlt_head = np.nan
        self.spin = 0

    def navigate(self, xyz, roll_pitch_yaw):
        yaw = roll_pitch_yaw[2]

        dlt_loc = self.xy_goal - xyz[0:2]
        trg_head = np.arctan2(dlt_loc[1], dlt_loc[0])

        _dlt_head = trg_head - yaw
        dlt_head = np.arctan2(np.sin(_dlt_head), np.cos(_dlt_head))
        if np.isnan(self.prev_dlt_head):
            self.prev_dlt_head = dlt_head

        if np.abs(self.prev_dlt_head) > np.pi/2 and np.sign(self.prev_dlt_head) != np.sign(dlt_head):
            # If prev dlt was in back and the signs just switched then the robot just spun
            self.spin += np.sign(self.prev_dlt_head)
            self.spin = np.clip(self.spin, a_min=-1, a_max=1)

        if self.spin == 0:
            turn_vel = dlt_head
            forward_vel = np.maximum(np.linalg.norm(dlt_loc) * np.cos(dlt_head), 0)
        else:
            turn_vel = self.spin * np.pi
            forward_vel = 0

        # norming
        forward_vel = np.tanh(forward_vel) * self.max_linear_velocity
        turn_vel = np.tanh(turn_vel) * self.max_turn_velocity

        # zero collapse
        if np.abs(forward_vel) < self.zero_collapse_epsilon:
            forward_vel = 0.
        if np.abs(turn_vel) < self.zero_collapse_epsilon:
            turn_vel = 0.
        self.prev_dlt_head = dlt_head
        return forward_vel, 0., turn_vel


def motion_target(sensory_dimension, granularity,
                  vel_head_val: float, vel_side_val: float, vel_direrction_val: float,
                          vel_w=1., ang_w=0.1, ang_z_w=1., eff_w=0.01, vel_side_w=None,
                          vel_direction_metric_bounded=True, vel_head_metric_bounded=True
                          ):
    """heading_vel, roll_vel, pitch_vel, yaw_vel"""
    forward_sense = 0
    direction_sense = 3
    side_sense = 4
    if vel_side_w is None:
        vel_side_w = vel_w

    value = np.zeros((sensory_dimension, granularity))
    weight = np.zeros((sensory_dimension, granularity))
    metric = np.zeros((sensory_dimension, granularity))

    # Twists and efforts have target value 0 with metric 0
    weight[1:3, :] = ang_w
    weight[5:, :] = eff_w

    # Forward, sideways and turning velocities depend on input
    value[forward_sense, :] = vel_head_val
    value[direction_sense, :] = vel_direrction_val
    value[side_sense, :] = vel_side_val

    weight[forward_sense, :] = vel_w
    weight[side_sense, :] = vel_side_w
    weight[direction_sense, :] = ang_z_w

    if vel_head_metric_bounded:
        metric[forward_sense, :] = -np.sign(vel_head_val)
    if vel_direction_metric_bounded:
        metric[direction_sense, :] = -np.sign(vel_direrction_val)
    # metric[side_sense, :] = -np.sign(vel_side_val)

    return EmbeddedTargetParameter(value=value, weight=weight, metric=metric)