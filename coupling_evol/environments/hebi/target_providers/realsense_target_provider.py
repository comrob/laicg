import numpy as np

from coupling_evol.engine.embedded_control import EmbeddedTargetParameter
from coupling_evol.engine.experiment_executor import TargetProvider
from coupling_evol.environments.hebi.realsense_listener import RealSenseListener, position_and_heading
import coupling_evol.engine.common as C

import coupling_evol.data_process.inprocess.recorder as rlog
from coupling_evol.engine import common as C
REC = rlog.get_recorder(C.RecordNamespace.TARGET_PROVIDER.key)


R_FORWARD_VEL_REF = "vel_ref_fwd"
R_SIDE_VEL_REF = "vel_ref_side"
R_TURN_VELOCITY_REF = "vel_ref_trn"
R_POS_XYZ = "pos_xyz"
R_POS_RPY = "pos_rpy"
R_GOAL_XY = "goal_xy"




class RealsenseTargetProvider(TargetProvider):
    def __init__(self, sensory_dim, granularity, xy_goal=(0, 0), max_linear_velocity=2., max_turn_velocity=1.,
                 zero_collapse_epsilon=0.01, vel_w=1., ang_z_w=1., ang_w=0.1, eff_w=0.1):
        super().__init__()
        self.sensory_dim = sensory_dim
        self.granularity = granularity
        self.xy_goal = np.asarray(xy_goal)
        self.max_linear_velocity = max_linear_velocity
        self.max_turn_velocity = max_turn_velocity
        self.zero_collapse_epsilon = zero_collapse_epsilon
        self.vel_w = vel_w
        self.ang_z_w = ang_z_w
        self.ang_w = ang_w
        self.eff_w = eff_w

    @staticmethod
    def get_data():
        return RealSenseListener.get_last_instance().data

    def __call__(self, **kwargs) -> EmbeddedTargetParameter:
        pass


class Translate(RealsenseTargetProvider):
    """
    Should keep same heading but crawl there sideways or forward.
    """

    def navigate(self, xyz, roll_pitch_yaw):
        dlt_loc = self.xy_goal - xyz[0:2]
        trg_head = np.arctan2(dlt_loc[1], dlt_loc[0])
        dlt_head = trg_head - roll_pitch_yaw[2]
        dist = np.linalg.norm(dlt_loc)
        forward_vel = dist * np.cos(dlt_head)
        side_vel = dist * np.sin(dlt_head)

        # zero collapse
        if np.abs(forward_vel) < self.zero_collapse_epsilon:
            forward_vel = 0.

        if np.abs(side_vel) < self.zero_collapse_epsilon:
            side_vel = 0.

        return forward_vel, side_vel, 0.

    def __call__(self, data: np.ndarray):
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
            vel_w=self.vel_w, ang_z_w=self.ang_z_w, ang_w=self.ang_w, eff_w=self.eff_w
        )


class TurnAndGo(RealsenseTargetProvider):
    """
    Should turn its head towards the goal and go forward at the same time.
    """

    def navigate(self, xyz, roll_pitch_yaw):
        dlt_loc = self.xy_goal - xyz[0:2]
        trg_head = np.arctan2(dlt_loc[1], dlt_loc[0])
        dlt_head = trg_head - roll_pitch_yaw[2]

        forward_vel = np.linalg.norm(dlt_loc) * np.cos(dlt_head)
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

    def __call__(self, data: np.ndarray):
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
            vel_w=self.vel_w, ang_z_w=self.ang_z_w, ang_w=self.ang_w, eff_w=self.eff_w
        )


def motion_target(sensory_dimension, granularity,
                  vel_head_val: float, vel_side_val: float, vel_direrction_val: float,
                  vel_w=1., ang_w=0.1, ang_z_w=1., eff_w=0.01):
    """heading_vel, roll_vel, pitch_vel, yaw_vel"""
    forward_sense = 0
    direction_sense = 3
    side_sense = 4

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
    weight[side_sense, :] = vel_w
    weight[direction_sense, :] = ang_z_w

    metric[forward_sense, :] = -np.sign(vel_head_val)
    metric[direction_sense, :] = -np.sign(vel_direrction_val)
    metric[side_sense, :] = -np.sign(vel_side_val)

    return EmbeddedTargetParameter(value=value, weight=weight, metric=metric)


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    ##
    rsl = RealSenseListener()
    rsl.listen_start()
    ##
    # trg = TurnAndGo(23, 6, zero_collapse_epsilon=0.1)
    trg = Translate(23, 6)
    ##
    data = []
    embs = []
    for i in range(3000):
        data.append(rsl.get_sensory_observation())
        t = trg()
        embs.append(t)
        print(f"fwd:{t.value[0, 0]:1.1f}[{t.metric[0, 0]:1.1f}], sid:{t.value[4, 0]:1.1f}[{t.metric[4, 0]:1.1f}],"
              f" trn:{t.value[3, 0]:1.1f}[{t.metric[3, 0]:1.1f}]")
        time.sleep(0.01)
    rsl.listen_end()
    ##
    xyz = np.asarray([position_and_heading(d)[0] for d in data])
    rpy = np.asarray([position_and_heading(d)[1] for d in data])
    vals = np.asarray([emb.value for emb in embs])
    mets = np.asarray([emb.metric for emb in embs])
    ##

    fig = plt.figure()
    fig.suptitle("Target")
    _f = fig.subplots(3, 1)
    for i, seg in enumerate([("fwd", 0), ("sid", 4), ("trn", 3)]):
        _f[i].plot(vals[:, seg[1], 0])
        _f[i].plot(mets[:, seg[1], 0])
        _f[i].set_ylabel(seg[0])

    ##
    fig = plt.figure()
    fig.suptitle("Odom lin XY and rot Z")
    _f = fig.subplots(3, 1)
    for i, seg in enumerate([("x", 0, xyz), ("y", 1, xyz), ("rotZ", 2, rpy)]):
        _f[i].plot(seg[2][:, seg[1]])
        _f[i].plot(seg[2][:, seg[1]])
        _f[i].set_ylabel(seg[0])

    ##
    fig = plt.figure()
    fig.suptitle("Map")
    _f = fig.subplots(1, 1)
    _f.plot(xyz[:, 0], xyz[:, 1])
    _f.plot([0.], [0.], 'bo')

    orig = []
    vec_fw = []
    vec_sid = []
    vec_head = []
    c_fw = []
    c_sid = []
    for i in range(0, len(data), 100):
        orig.append([xyz[i, 0], xyz[i, 1]])
        forward = vals[i, 0, 0]
        side = vals[i, 4, 0]
        turn = vals[i, 3, 0]
        head = rpy[i, 2]
        if forward > 0:
            c_fw.append('g')
        else:
            c_fw.append('r')

        if side > 0:
            c_sid.append('g')
        else:
            c_sid.append('r')

        vec_fw.append([np.cos(turn + head) * forward, np.sin(turn + head) * forward])
        vec_sid.append([np.sin(turn + head) * side, np.cos(turn + head) * side])
        vec_head.append([np.cos(head), np.sin(head)])

    orig = np.asarray(orig)
    vec_fw = np.asarray(vec_fw) / 10
    _f.quiver(orig[:, 0], orig[:, 1], vec_fw[:, 0], vec_fw[:, 1], color=c_fw)
    vec_sid = np.asarray(vec_sid) / 10
    _f.quiver(orig[:, 0], orig[:, 1], vec_sid[:, 0], vec_sid[:, 1], color=c_sid)
    vec_head = np.asarray(vec_head) / 10
    _f.quiver(orig[:, 0], orig[:, 1], vec_head[:, 0], vec_head[:, 1], color=['k'] * len(c_sid))
    ##
    plt.show()
