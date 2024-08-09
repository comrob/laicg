# if __name__ == "__main__":
#     import sys
#     import os
#     sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#     from hexapod_sim import RobotHAL
#     from hexapod_sim.robot_consts import *
#     from environment import *
# else:
import numpy as np

from coupling_evol.environments.coppeliasim.hexapod_sim.robot_consts import *
from coupling_evol.engine.environment import *
from coupling_evol.environments.utils.filter import *
from typing import List, Union
import coupling_evol.data_process.inprocess.recorder as rlog
from coupling_evol.engine import common as C
# REC = rlog.get_recorder(C.RecordNamespace.SENSORY_SOURCE.key)
REC = rlog.get_null_recorder()

##################### Brief description in words and pictures:
# body (b)
# _____
#    __|_coxa (c)
#   /    \___________
#  |   |   femur (f) \
#  |   |  ________    \
#   \____/        \    \ tibia (t)
# _____|           \    \                  
#                   \   /
#                    \_/
#
#  bc = body-coxa joint (maximum forward = 1, maximum backward = -1)
#  cf = coxa-femur joint (maximum up = 1, maximum down = -1)
#  ft = femur-tibia joint (maximum up = 1, maximum down = -1)
#
# Matrix 'U' of shape (3, 6) (used as argument u = U.T in fuction
# CoppeliaSimSession.step) is expected as follows:
#    leg 0: [bc, cf, ft]
#    leg 1: [bc, cf, ft]
#    leg 2: [bc, cf, ft]
#    leg 3: [bc, cf, ft]
#    leg 4: [bc, cf, ft]
#    leg 5: [bc, cf, ft]
#
#   where legs numbering is following:
#                  /\
#                  ||
#       <--o--o-- 0___1--o--o-->
#                /     \
#      <--o--o--4       5--o--o-->
#                \     /
#        <--o--o--2___3--o--o-->
#
# CoppeliaSimSession.step takes values in range [-1, 1] in format of matrix 'u'
# 6x3 (or the flattened version of such matrix) and maps them to correct
# robot servos with correct corresponding values.
# 
# ampl_max and ampl_min in CoppeliaSimEnvironment give the maximum and minimal
# amplitudes, respectively, for the joints [bc, cf, ft] in radian units
# (i.e. rotation of 360 degrees is 2pi).
# The final servo value/position/angle is then given as:
# 
#       pos = (dps + v * ampl) * pn
#
# where pos  = postion to be send to the servo
#       dps  = defult position of the servo given by DEFAULT_POS values in 
#              robot_consts.py
#       v    = value given by matrix 'u' for the joint (in range [-1, 1])
#       ampl = ampl_max for given joint, if v in [0, 1], or 
#              ampl_min for given joint, if v in [-1, 0]
#       pn   = constant 1 (positive) or -1 (negative) depending on the setting
#              of the servo in robot. This unifies the range of values.
#              By default, some servos on the robot are set to move up/forward
#              if set to positive and down/backward if the position is set to
#              negative, while some other servos work the other way around.
#              This constant ensures, that for all servos the positive and
#              negative values have the same meaning.
#
# The values are then mapped from the format of matrix 'u' to the format
# of the robot setting:
# 
#                   /\
#                   ||
#     <-- 4-- 2-- 0___1 -- 3-- 5-->
#                /     \
#   <--10-- 8-- 6       7 -- 9--11-->
#                \     /
#     <--16--14--12___13--15--17-->

REMAP = [0, 2, 4, 1, 3, 5, 12, 14, 16, 13, 15, 17, 6, 8, 10, 7, 9, 11]

_RIGHT_ANGLE = np.pi / 2
Y_DIMENSION = 23
XYZ_DIMENSION = 3
Q_DIMENSION = 4

def get_heading(q):
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/index.htm
    # arcsin(2XY + 2ZW)
    _r = np.arcsin(2 * q[0] * q[1] + 2 * q[2] * q[3])
    # arctan2(2YW-2XZ, 1-2YY-2ZZ)
    _p = np.arctan2(2 * q[1] * q[3] - 2 * q[0] * q[2], 1 - 2 * q[1] * q[1] - 2 * q[2] * q[2]) + _RIGHT_ANGLE
    # arctan2(2XW-2YZ, 1-2XX-2ZZ)
    yaw = np.arctan2(2 * q[0] * q[3] - 2 * q[1] * q[2], 1 - 2 * q[0] * q[0] - 2 * q[2] * q[2]) + np.pi

    roll = np.cos(yaw) * _r - np.sin(yaw) * _p
    pitch = np.sin(yaw) * _r + np.cos(yaw) * _p
    return np.asarray([roll, pitch, yaw])


def heading_and_side_speeds(roll_pitch_yaw, velocity_vector):
    vel_dir = np.arctan2(velocity_vector[1], velocity_vector[0])
    vel_amp = np.sqrt(np.sum(np.square(velocity_vector[:2])))
    return [vel_amp * np.cos(vel_dir - roll_pitch_yaw[2]), vel_amp * np.cos(vel_dir - roll_pitch_yaw[2] + _RIGHT_ANGLE)]


class DifferentialHeadSideAngularEffortSpeedsProcessor:
    # head_speed, rotx, roty, rotz, side_speed
    RESCALE = np.asarray([.25 * .07, 0.25 * .5, .25 * .5, .25 * .5, .25 * .07] +
                         [3.] * 18)  # measured as std from babble mem

    # RESCALE = np.asarray([.25] * 23) #measured as std from babble mem

    def __init__(self, delta_t=0.01, clip=0.1, rescale=RESCALE, cheap_angle_diff_filter=True):
        # head_speed, rotx, roty, rotz, side_speed
        self.xyz_filter = DifferentialFilter(dimension=3, delta_t=delta_t, clip=clip)
        self.rpy_filter = AngleDifferentialFilter(dimension=3, delta_t=delta_t, clip=clip,
                                                  cheap_diff=cheap_angle_diff_filter)
        self.effort_filter = DifferentialFilter(dimension=18, delta_t=delta_t, clip=1)
        self.rescale = rescale

    def __call__(self, xyz, q, eff):
        ret = np.zeros((Y_DIMENSION,))
        rpy = get_heading(q)
        d_xyz = self.xyz_filter(xyz)
        d_rpy = self.rpy_filter(rpy)
        d_eff = self.effort_filter(eff)
        head_s, side_s = heading_and_side_speeds(rpy, d_xyz)
        ret[0] = head_s
        ret[1:4] = d_rpy
        ret[4] = side_s
        ret[5:] = d_eff
        return ret / self.rescale


class CoppeliaSimSession(Session):
    R_SIMTIME = "time"
    R_POS = "robt_pos"
    R_ORIENTATION = "ori"
    R_TORQUES = "tor"
    R_SERVO_POS = "servo_pos"
    R_Y_RAW = "y_raw"

    def __init__(self,
                 coefficient: np.ndarray,
                 offset: np.ndarray):
        super().__init__()
        from coupling_evol.environments.coppeliasim.hexapod_sim import RobotHAL
        self.rh = RobotHAL.RobotHAL()
        self.rh.get_sim_time()
        self.rh.get_all_servo_position()
        self.rh.get_all_joint_torques()
        self.rh.get_robot_position()
        self.rh.get_robot_orientation()

        self.coefficient = coefficient
        self.offset = offset

        self._xyz = np.zeros((XYZ_DIMENSION,))
        self._q = np.zeros((Q_DIMENSION,))
        self._eff = np.zeros((18,))

        self.raw_processor = DifferentialHeadSideAngularEffortSpeedsProcessor(
            delta_t=0.01, clip=3, rescale=DifferentialHeadSideAngularEffortSpeedsProcessor.RESCALE,
            cheap_angle_diff_filter=True)

        self.rh.set_all_servo_position_slow(DEFAULT_POS + self.offset, TIME_FRAME, 222 * TIME_FRAME)

    def step(self, u: np.ndarray) -> np.ndarray:
        _u = np.zeros((18,))
        _u[REMAP] = u
        _cmd = DEFAULT_POS + (self.coefficient * _u + self.offset)  # * POS_NEG
        self.rh.set_all_servo_position(_cmd)

        pos = self.rh.get_robot_position()  # 3
        if pos is not None:
            self._xyz = np.asarray(pos)
        q = self.rh.get_robot_quaternion()  # 4
        if q is not None:
            self._q = np.asarray(q)
        torque = (self.rh.get_all_joint_torques() * POS_NEG)  # 18
        if torque is not None:
            self._eff = np.asarray(torque)

        _y = self.raw_processor(self._xyz, self._q, self._eff)

        REC(self.R_POS, self._xyz)
        REC(self.R_ORIENTATION, self._q)
        REC(self.R_TORQUES, self._eff)

        y = np.zeros((Y_DIMENSION + XYZ_DIMENSION + Q_DIMENSION,))
        y[:Y_DIMENSION] = _y
        y[Y_DIMENSION:Y_DIMENSION+XYZ_DIMENSION] += self._xyz
        y[Y_DIMENSION+XYZ_DIMENSION:] += self._q

        return y

    def end(self):
        self.rh.set_all_servo_position_slow(DEFAULT_POS + self.offset * POS_NEG, TIME_FRAME, 222 * TIME_FRAME)
        self.rh.stop_simulation()


class CoppeliaSimEnvironment(Environment):
    def __init__(self,
                 ampl_min: Union[np.ndarray, List[float]],
                 ampl_max: Union[np.ndarray, List[float]]):
        super().__init__()
        ampl_min = np.asarray(ampl_min)
        ampl_max = np.asarray(ampl_max)
        self.scale, self.offset = self._get_ampl(ampl_min, ampl_max)
        self.relative_pos = np.zeros((3, 6))
        self.session = None

    @staticmethod
    def _get_ampl(ampl_min: np.array, ampl_max: np.array):
        assert ampl_min.shape == ampl_max.shape == (3,)
        offset = np.tile(((ampl_min + ampl_max) / 2)[:, np.newaxis], (1, 6))
        coef = ampl_max[:, np.newaxis] - offset
        return coef.flatten()[MAPPING], offset.flatten()[MAPPING] * POS_NEG

    def create_session(self) -> Session:
        self.session = CoppeliaSimSession(self.scale, self.offset)
        return self.session

    def end_session(self):
        self.session.end()

    def position_zero(self):
        if self.session is not None:
            self.session.rh.set_all_servo_position_slow(DEFAULT_POS + self.offset, TIME_FRAME, 222 * TIME_FRAME)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from coupling_evol.data_process.inprocess import records as R

    import coupling_evol.data_process.inprocess.record_logger as rlog
    rl = rlog.RecordAggregator()
    REC = rl.record

    CNST = CoppeliaSimSession

    # run_exp = False
    run_exp = True
    if run_exp:
        ampl_min = np.array([-0.32, 0.2, -0.2])
        ampl_max = np.array([0.32, 0.7, 0.3])

        env = CoppeliaSimEnvironment(ampl_min, ampl_max)
        ses = env.create_session()

        cyc_num = 2000
        rec_raw = []

        print("going to position zero")
        env.position_zero()
        print("session starts")
        for i in range(cyc_num):
            _cmd = np.zeros(18)
            wave = np.sin(i * 0.01 * (2 * np.pi)) * .5
            if i < cyc_num * 0.15:
                # Shoulder (femur) - Z
                print("LIN Z")
                _cmd[[0, 6, 12]] = 0
                _cmd[[3, 9, 15]] = 0
                _cmd[[1, 7, 13]] = wave
                _cmd[[4, 10, 16]] = -wave
            elif i < cyc_num * 0.3:
                # Base (coxas) - X
                print("LIN X")
                _cmd[[0, 6, 12]] = wave
                _cmd[[3, 9, 15]] = -wave
            elif i < cyc_num * 0.45:
                # Shoulder (femur) - Y
                print("LIN Y")
                _cmd[[1, 7, 13]] = 0
                _cmd[[4, 10, 16]] = 0
                _cmd[[2, 8, 14]] = wave
                _cmd[[5, 11, 17]] = wave
            elif i < cyc_num * 0.6:
                # Shoulder (femur) - rot Z
                print("ROT Z")
                _cmd[[0, 6, 12]] = wave
                _cmd[[3, 9, 15]] = wave
                _cmd[[1, 7, 13]] = 0
                _cmd[[4, 10, 16]] = 0
                _cmd[[2, 8, 14]] = 0
                _cmd[[5, 11, 17]] = 0
            if i < cyc_num * 0.5:
                # Shoulder (femur) - rot X  - ROLL
                print("ROT X")
                _cmd[[0, 6, 12]] = 0
                _cmd[[3, 9, 15]] = 0
                _cmd[[1, 7, 13]] = wave * 1
                _cmd[[4, 10, 16]] = wave * 1
                _cmd[[2, 8, 14]] = 0
                _cmd[[5, 11, 17]] = 0
            else:
                # Shoulder (femur) - rot Y
                print("ROT Y")
                _cmd[[0, 6, 12]] = 0
                _cmd[[3, 9, 15]] = 0
                _cmd[[1, 16]] = wave * 1
                _cmd[[4, 13]] = -wave * 1
                _cmd[[2, 8, 14]] = 0
                _cmd[[5, 11, 17]] = 0

            # wave = np.sin(i * 0.01 * (2 * np.pi)) * 5
            # _cmd[[0, 6, 12]] = 0
            # _cmd[[3, 9, 15]] = 0
            # _cmd[[1, 16]] = wave
            # _cmd[[4, 13]] = -wave
            # _cmd[[2, 8, 14]] = 0
            # _cmd[[5, 11, 17]] = 0
            try:
                print(f"{i}/{cyc_num}")
                y = ses.step(_cmd)
                rl.increment()
                row = rl.get_last_record()
                row["y"] = y
                rec_raw.append(row)
                time.sleep(1 / 100.)
            except Exception as e:
                print(e)
                env.end_session()
                exit(-1)
        env.end_session()
        R.save_records("rec.hdf5", R.numpyfy_record(rec_raw))

    r = R.load_records("rec.hdf5")[0]
    R.print_record_shapes(r)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_title("Position")
    ax.scatter3D(r[CNST.R_POS][:, 0], r[CNST.R_POS][:, 1], r[CNST.R_POS][:, 2])

    plt.rcParams["figure.figsize"] = (15, 5)
    fig = plt.figure()
    fig.suptitle("X,Y,Z")
    _f = fig.subplots(3, 1)
    _f[0].plot(r[CNST.R_POS][:, 0])
    _f[1].plot(r[CNST.R_POS][:, 1])
    _f[2].plot(r[CNST.R_POS][:, 2])

    plt.rcParams["figure.figsize"] = (15, 5)
    fig = plt.figure()
    fig.suptitle("ROLL,PITCH,YAW")
    _f = fig.subplots(3, 1)
    rpy = np.asarray([get_heading(q) for q in r[CNST.R_ORIENTATION]])
    _f[0].plot(rpy[10:, 0])
    _f[1].plot(rpy[10:, 1])
    _f[2].plot(rpy[10:, 2])

    plt.rcParams["figure.figsize"] = (15, 5)
    fig = plt.figure()
    fig.suptitle("y_output")
    _f = fig.subplots(5, 1)
    for i, dim in enumerate(
            ["heading_vel", "roll_vel", "pitch_vel", "yaw_vel", "side_vel"]):  # + [f"f[{i}]" for i in range(18)]):
        rc = r["y"][60:, i]
        rc_m = np.mean(rc)
        rc_s = np.std(rc)
        print(f"Stats {dim}: {rc_m}({rc_s})")

        f = _f[i]
        f.plot(rc)
        f.axhline(y=rc_m, color='k', linestyle='--')
        f.axhline(y=rc_m - rc_s, color='g', linestyle='--')
        f.axhline(y=rc_m + rc_s, color='g', linestyle='--')

        f.set_ylabel(f"{dim}")
    ##
    plt.savefig("y_output.png")
    plt.show()
