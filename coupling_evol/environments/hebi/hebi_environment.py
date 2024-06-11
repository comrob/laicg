from coupling_evol.environments.hebi.robot_interface import Lily, CommandAndFeedback, get_symmetric_joint_command, Daisy
# from environments.hebi.rig_topic_listener import RigTopicListener
from coupling_evol.environments.hebi.realsense_listener import RealSenseListener, FakeSenseListener, extract_heading_speed, extract_angular_velocity, DATA_LEN, ID_ROTATION, ID_TRANSLATION
from coupling_evol.environments.hebi.realsense_listener import SumTemporalDerivativeFilter, extract_heading_and_side_speeds
from coupling_evol.environments.hebi.realsense_listener import position_and_heading, heading_and_side_speeds
import time
import logging
from coupling_evol.engine.environment import *
from typing import Union
LOG = logging.getLogger(__name__)
from enum import Enum
from coupling_evol.environments.utils.filter import *

import coupling_evol.data_process.inprocess.recorder as rlog
from coupling_evol.engine import common as C
# REC = rlog.get_recorder(C.RecordNamespace.SENSORY_SOURCE.key)
REC = rlog.get_null_recorder()

HRPYS_DIMENSION = 5
EFF_DIMENSION = 18
XYZ_DIMENSION = 3
Q_DIMENSION = 4

class SenseType(Enum):
    REAL = RealSenseListener,
    FAKE = FakeSenseListener

    def __init__(self, clazz):
        self.clazz = clazz


class HebiType(Enum):
    DAISY = Daisy,
    LILY = Lily

    def __init__(self, clazz):
        self.clazz = clazz


class HeadSpeedAndAngularVelocityProcessor:
    def __init__(self, filter_window):
        # head_speed, rotx, roty, rotz
        self.filter = SumTemporalDerivativeFilter(sample_length=filter_window, dimension=5)

    def __call__(self, y_raw):
        y = get_head_speed_and_angular_velocity(y_raw)
        return self.filter(y)


class MeanDifferentialHeadSideAngularSpeedsProcessor:
    # head_speed, rotx, roty, rotz, side_speed
    RESCALE = np.asarray([0.056, 0.151, 0.141, 0.10, 0.067]) #measured as std from ENV rep
    def __init__(self, delta_t=0.01, sample_length=10, clip=0.1, rescale=RESCALE, angle_diff_filter=False,
                 cheap_angle_diff_filter=True
                 ):
        # head_speed, rotx, roty, rotz, side_speed
        self.xyz_filter = DifferentialFilter(dimension=3, delta_t=delta_t, clip=clip)
        if angle_diff_filter:
            self.rpy_filter = AngleDifferentialFilter(dimension=3, delta_t=delta_t, clip=clip,
                                                      cheap_diff=cheap_angle_diff_filter)
        else:
            self.rpy_filter = DifferentialFilter(dimension=3, delta_t=delta_t, clip=clip)
        self.xyz_filter_m = MeanFilter(dimension=3, sample_length=sample_length)
        self.rpy_filter_m = MeanFilter(dimension=3, sample_length=sample_length)
        self.aggregate_xyz = self.xyz_filter_m
        self.aggregate_rpy = self.rpy_filter_m
        self.rescale = rescale

    def __call__(self, y_raw):
        ret = np.zeros((5, ))
        xyz, rpy = position_and_heading(sensory_observation=y_raw)
        _d_xyz = self.xyz_filter(xyz)
        _d_rpy = self.rpy_filter(rpy)
        d_xyz = self.xyz_filter_m(_d_xyz)
        d_rpy = self.rpy_filter_m(_d_rpy)
        head_s, side_s = heading_and_side_speeds(rpy, d_xyz)

        ret[0] = head_s
        ret[1:4] = d_rpy
        ret[4] = side_s
        return ret/self.rescale


class DifferentialHeadSideAngularSpeedsProcessor:
    # head_speed, rotx, roty, rotz, side_speed
    RESCALE = np.asarray([0.056, 0.151, 0.141, 0.10, 0.067]) #measured as std from ENV rep
    def __init__(self, delta_t=0.01, clip=0.1, rescale=RESCALE,
                 cheap_angle_diff_filter=True):
        # head_speed, rotx, roty, rotz, side_speed
        self.xyz_filter = DifferentialFilter(dimension=3, delta_t=delta_t, clip=clip)
        self.rpy_filter = AngleDifferentialFilter(dimension=3, delta_t=delta_t, clip=clip,
                                                      cheap_diff=cheap_angle_diff_filter)
        self.rescale = rescale

    def __call__(self, y_raw):
        ret = np.zeros((5, ))
        xyz, rpy = position_and_heading(sensory_observation=y_raw)
        d_xyz = self.xyz_filter(xyz)
        d_rpy = self.rpy_filter(rpy)
        head_s, side_s = heading_and_side_speeds(rpy, d_xyz)
        ret[0] = head_s
        ret[1:4] = d_rpy
        ret[4] = side_s
        return ret/self.rescale


class DifferentialHeadSideAngularSpeedsEffortProcessor:
    # head_speed, rotx, roty, rotz, side_speed
    # RESCALE = np.asarray([0.056, 0.151, 0.141, 0.10, 0.067] + [1.] * 1) #measured as std from ENV rep
    RESCALE = np.asarray([0.056, 0.05, 0.05, 0.05, 0.067] + [1., 3., 2.] * 6) #measured as std from ENV rep
    def __init__(self, delta_t=0.01, clip=0.1, rescale=RESCALE,
                 cheap_angle_diff_filter=True):
        # head_speed, rotx, roty, rotz, side_speed
        self.xyz_filter = DifferentialFilter(dimension=3, delta_t=delta_t, clip=clip)
        self.rpy_filter = AngleDifferentialFilter(dimension=3, delta_t=delta_t, clip=clip,
                                                      cheap_diff=cheap_angle_diff_filter)
        self.rescale = rescale

    def __call__(self, y_raw):
        # ret = np.zeros((6, ))
        ret = np.zeros((23, ))
        xyz, rpy = position_and_heading(sensory_observation=y_raw[:13])
        d_xyz = self.xyz_filter(xyz)
        d_rpy = self.rpy_filter(rpy)
        head_s, side_s = heading_and_side_speeds(rpy, d_xyz)
        ret[0] = head_s
        ret[1:4] = d_rpy
        ret[4] = side_s
        # ret[5] = np.sum(y_raw[13:])
        ret[5:] = y_raw[13:]
        return ret/self.rescale


class DifferentialHeadSideAngularSpeedsSumAbsEffortProcessor:
    # head_speed, rotx, roty, rotz, side_speed
    RESCALE = np.asarray([0.05, 0.10, 0.10, 0.10, 0.05] + [0.3] * 1) #measured as std from ENV rep
    def __init__(self, delta_t=0.01, clip=0.1, rescale=RESCALE,
                 cheap_angle_diff_filter=True):
        # head_speed, rotx, roty, rotz, side_speed
        self.xyz_filter = DifferentialFilter(dimension=3, delta_t=delta_t, clip=clip)
        self.rpy_filter = AngleDifferentialFilter(dimension=3, delta_t=delta_t, clip=clip,
                                                      cheap_diff=cheap_angle_diff_filter)
        self.rescale = rescale

    def __call__(self, y_raw):
        ret = np.zeros((6, ))
        xyz, rpy = position_and_heading(sensory_observation=y_raw[:13])
        d_xyz = self.xyz_filter(xyz)
        d_rpy = self.rpy_filter(rpy)
        head_s, side_s = heading_and_side_speeds(rpy, d_xyz)
        ret[0] = head_s
        ret[1:4] = d_rpy
        ret[4] = side_s
        ret[5] = np.sum(np.abs(y_raw[13:]))
        return ret/self.rescale


class DifferentialHeadSideAngularEffortSpeedsProcessor:
    # head_speed, rotx, roty, rotz, side_speed
    # RESCALE = np.asarray([0.056, 0.151, 0.141, 0.10, 0.067] + [1.] * 1) #measured as std from ENV rep
    # RESCALE = np.asarray([0.056, 0.05, 0.05, 0.05, 0.067] + [15., 24., 18.] * 6) #measured as std from babble
    RESCALE = np.asarray([0.05, 0.05, 0.05, 0.04, 0.05] + [8., 15., 10.] * 6) #measured as std from babble mem
    def __init__(self, delta_t=0.01, clip=0.1, rescale=RESCALE,
                 cheap_angle_diff_filter=True):
        # head_speed, rotx, roty, rotz, side_speed
        self.xyz_filter = DifferentialFilter(dimension=3, delta_t=delta_t, clip=clip)
        self.rpy_filter = AngleDifferentialFilter(dimension=3, delta_t=delta_t, clip=clip,
                                                      cheap_diff=cheap_angle_diff_filter)
        self.effort_filter = DifferentialFilter(dimension=18, delta_t=delta_t, clip=1)

        self.rescale = rescale

    def __call__(self, y_raw):
        # ret = np.zeros((6, ))
        ret = np.zeros((23, ))
        xyz, rpy = position_and_heading(sensory_observation=y_raw[:13])
        d_xyz = self.xyz_filter(xyz)
        d_rpy = self.rpy_filter(rpy)
        d_eff = self.effort_filter(y_raw[13:])
        head_s, side_s = heading_and_side_speeds(rpy, d_xyz)
        ret[0] = head_s
        ret[1:4] = d_rpy
        ret[4] = side_s
        ret[5:] = d_eff
        return ret/self.rescale

class HebiEnvironmentSession(Session):
    U_INPUT_NAME = "u_in"
    Y_OUTPUT_NAME = "y_out"

    U_NORMED = "u_nrm"
    U_POS_REF = "u_pos_ref"

    U_HEBI_POS = "u_hebi_pos"
    U_HEBI_EFF = "u_hebi_eff"
    Y_E_RAW = "y_e_raw"

    Y_P_EFFORT = "y_p_eff"
    Y_P_POSITION = "y_p_pos"

    def __init__(self, topic_listener: Union[RealSenseListener, FakeSenseListener], caf: CommandAndFeedback,
                 robot_type: HebiType = HebiType.LILY, position_strategy=True):
        # params
        ## init topic listener
        super().__init__()
        self.topic_listener = topic_listener
        vicon_ok = False
        for i in range(10):
            try:
                _ = self.topic_listener.get_sensory_observation()
                vicon_ok = True
            except Exception as e:
                LOG.warning("Rig listener doesn't provide. Trial {}/10".format(i + 1))
                time.sleep(5)
            if vicon_ok:
                break
        if not vicon_ok:
            LOG.error("Rig does not work, shutting down.")
            exit(-1)
        ## init serial link
        self.caf = caf
        ## history
        self.position_strategy = position_strategy
        self.cou = 0
        self.u_hebi_pos = np.zeros(18)
        self.u_hebi_eff = np.zeros(18)
        self.robot_class = robot_type.clazz
        self.raw_processor = DifferentialHeadSideAngularEffortSpeedsProcessor(
            delta_t=0.01, clip=0.1, rescale=DifferentialHeadSideAngularEffortSpeedsProcessor.RESCALE,
            cheap_angle_diff_filter=True)

    def step(self, u: np.ndarray) -> np.ndarray:
        ##
        u_nrm = np.clip(u, a_min=-1, a_max=1)
        pos_ref = self.robot_class.de_norm(u_nrm)
        # TODO maybe would be better to use pos ref strat and setup the PID controller...

        # effort control strategy
        self.caf.update_feedback()
        pos_curr = self.caf.get_position_feedback()
        eff_curr = self.caf.get_effort_feedback()

        if self.position_strategy:
            self.u_hebi_pos = pos_ref
            self.u_hebi_eff = np.zeros(pos_ref.shape) + np.nan
        else:
            efforts = effort_control(pos_ref, pos_curr, self.robot_class.JOINT_STIFFNESS)
            self.u_hebi_pos = np.zeros(efforts.shape) + np.nan
            self.u_hebi_eff = efforts

        # comanding robot
        self.caf.send_command(positions=self.u_hebi_pos, efforts=self.u_hebi_eff)
        ##
        y_raw = np.zeros((DATA_LEN + EFF_DIMENSION, ))
        y_raw[:DATA_LEN] = self.topic_listener.get_sensory_observation()
        y_raw[DATA_LEN:] = eff_curr
        # _y = self.raw_processor(y_raw)
        y = np.zeros((
            HRPYS_DIMENSION + EFF_DIMENSION + XYZ_DIMENSION + Q_DIMENSION,))
        y[:HRPYS_DIMENSION + EFF_DIMENSION] = self.raw_processor(y_raw)
        y[HRPYS_DIMENSION + EFF_DIMENSION:] = y_raw[ID_TRANSLATION:ID_ROTATION + Q_DIMENSION]

        REC(self.Y_E_RAW, y_raw)
        REC(self.Y_P_EFFORT, eff_curr)
        REC(self.Y_P_POSITION, pos_curr)
        REC(self.U_NORMED, u_nrm)
        REC(self.U_POS_REF, pos_ref)
        REC(self.U_HEBI_POS, self.u_hebi_pos)
        REC(self.U_HEBI_EFF, self.u_hebi_eff)
        REC(self.U_INPUT_NAME, np.zeros(u.shape) + u)
        REC(self.Y_OUTPUT_NAME, np.zeros(y.shape) + y)

        return y


class HebiEnvironment(Environment):
    def __init__(self, robot_type: HebiType = HebiType.LILY, sense_type: SenseType = SenseType.REAL):
        ## init vicon
        self.topic_listener = sense_type.clazz()
        self.topic_listener.listen_start()
        ## init robot
        self.robot_type = robot_type
        self.robot_class = robot_type.clazz
        self.robot = self.robot_class()
        self.caf = self.robot.init_command_and_feedback()

    def position_zero(self):
        self.robot.stand_up(self.caf)
        self.robot_class.safe_fix_position(self.caf,
                               get_symmetric_joint_command(self.robot_class.STANDING_LEG_POSITION),
                               safety_eps=0.4,
                               cycles=600)

    def end_session(self):
        self.robot.lie_down(self.caf)
        # self.topic_listener.listen_end()

    def create_session(self) -> HebiEnvironmentSession:
        return HebiEnvironmentSession(self.topic_listener, self.caf, robot_type=self.robot_type)


def get_position_angular_velocity(y_raw):
    return y_raw[7:]


def get_head_speed_and_angular_velocity(realsense_raw):
    ret = np.zeros((4,))
    ret[0] = extract_heading_speed(realsense_raw)
    ret[1:] = extract_angular_velocity(realsense_raw)
    return ret


def get_planar_speeds_and_angular_velocity(realsense_raw):
    ret = np.zeros((5,))
    pln_sp = extract_heading_and_side_speeds(realsense_raw)
    ret[0] = pln_sp[0]
    ret[1:4] = extract_angular_velocity(realsense_raw)
    ret[4] = pln_sp[1]
    return ret


def effort_control(pos_ref, pos_curr, stiffness):
    return (pos_ref - pos_curr) * stiffness


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
    import matplotlib.pyplot as plt
    from coupling_evol.data_process.inprocess import records as R

    import coupling_evol.data_process.inprocess.record_logger as rlog
    rl = rlog.RecordAggregator()
    REC = rl.record

    HE = HebiEnvironmentSession
    just_plots = False
    # just_plots = True

    if not just_plots:
        env = HebiEnvironment(robot_type=HebiType.DAISY, sense_type=SenseType.REAL)
        session = env.create_session()
        hist = []
        cyc_num = 3000

        print("going to position zero")
        env.position_zero()
        print("session starts")
        for i in range(cyc_num):
            _cmd = np.zeros(18)
            wave = np.sin(i * 0.01 * (2 * np.pi)) *.5
            if i % 10 == 0:
                print(i)
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
            elif i < cyc_num * 0.75:
                # Shoulder (femur) - rot X
                print("ROT X")
                _cmd[[0, 6, 12]] = 0
                _cmd[[3, 9, 15]] = 0
                _cmd[[1, 7, 13]] = wave
                _cmd[[4, 10, 16]] = wave
                _cmd[[2, 8, 14]] = 0
                _cmd[[5, 11, 17]] = 0
            else:
                # Shoulder (femur) - rot Y
                print("ROT Y")
                _cmd[[0, 6, 12]] = 0
                _cmd[[3, 9, 15]] = 0
                _cmd[[1, 16]] = wave
                _cmd[[4, 13]] = -wave
                _cmd[[2, 8, 14]] = 0
                _cmd[[5, 11, 17]] = 0

            try:
                session.step(_cmd)
                rl.increment()
                # time.sleep(1/session.caf.group.feedback_frequency)
                time.sleep(1 / 100.)
            except Exception as e:
                print(e)
                env.end_session()
                exit(-1)
        env.end_session()

        rec = R.numpyfy_record(rl.buffer)
        R.print_record_shapes(rec)
        R.save_records("env_data.hdf5", rec)

    r = R.load_records("env_data.hdf5")[0]
    R.print_record_shapes(r)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_title("Position")
    ax.scatter3D(r[HE.Y_E_RAW][:, 0], r[HE.Y_E_RAW][:, 1], r[HE.Y_E_RAW][:, 2])


    ##
    fig = plt.figure()
    fig.suptitle(HE.U_INPUT_NAME)
    _f = fig.subplots(6, 3)
    for j in range(6):
        f = _f[j]
        for i in range(3):
            f[i].plot(r[HE.U_INPUT_NAME][:, i + j * 3])
            if j == 0:
                f[i].set_title(str(i))
        f[0].set_ylabel(f"L[{j}]")
    plt.savefig("u_input.png")
    ##
    ##
    fig = plt.figure()
    fig.suptitle("Position reference and observed")
    _f = fig.subplots(6, 3)
    for j in range(6):
        f = _f[j]
        for i in range(3):
            f[i].plot(r[HE.U_POS_REF][:, i + j * 3], label=HE.U_POS_REF, alpha=0.7)
            f[i].plot(r[HE.Y_P_POSITION][:, i + j * 3], label=HE.Y_P_POSITION, alpha=0.7)
            if j == 0:
                f[i].set_title(str(i))
        f[0].set_ylabel(f"L[{j}]")
    _f[0][0].legend()
    plt.savefig("position_reference_and_observed.png")
    ##

    ##
    fig = plt.figure()
    fig.suptitle("Effort reference and observed")
    _f = fig.subplots(6, 3)
    for j in range(6):
        f = _f[j]
        for i in range(3):
            f[i].plot(r[HE.U_HEBI_EFF][:, i + j * 3], label=HE.U_HEBI_EFF, alpha=0.7)
            f[i].plot(r[HE.Y_P_EFFORT][:, i + j * 3], label=HE.Y_P_EFFORT, alpha=0.7)
            if j == 0:
                f[i].set_title(str(i))
        f[0].set_ylabel(f"L[{j}]")
    _f[0][0].legend()
    plt.savefig("effort_reference_and_observed.png")

    ##

    ##
    plt.rcParams["figure.figsize"] = (15, 5)
    fig = plt.figure()
    fig.suptitle(HE.Y_OUTPUT_NAME)
    _f = fig.subplots(5, 1)
    # for i, dim in enumerate(["heading_vel", "roll_vel", "pitch_vel", "yaw_vel", "side_vel"] + [f"f[{i}]" for i in range(18)]):
    for i, dim in enumerate(["heading_vel", "roll_vel", "pitch_vel", "yaw_vel", "side_vel"]):

        rc = r[HE.Y_OUTPUT_NAME][40:, i]
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



    ##

