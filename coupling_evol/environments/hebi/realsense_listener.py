import pyrealsense2 as rs
# rs = None
import time
import numpy as np
import logging

LOG = logging.getLogger(__name__)
ID_TRANSLATION = 0
ID_ROTATION = 3
ID_VELOCITY = 7
ID_ANGULAR_VELOCITY = 10
DATA_LEN = 13
_RIGHT_ANGLE = np.pi/2
_ROUND_ANGLE = np.pi * 2
REALSENSE_ROBOT_TRANSFER = [
    [0, -1, 0],
    [0, 0, 1],
    [-1, 0, 0]
]

class FakeSenseListener:
    def __init__(self):
        self.pipe = None
        self.cfg = None
        self.data = []

    def listen_start(self):
        LOG.info("The FAKESense pipe started")


    def listen_end(self):
        LOG.info("FAKESense pipe stopped.")


    def get_sensory_observation(self):
        return np.zeros((13,))


class RealSenseListener:
    _last_instance = None

    def __init__(self):
        self.pipe = None
        self.cfg = None
        self.data = []
        self.__class__._last_instance = self

    @classmethod
    def get_last_instance(cls):
        return cls._last_instance

    def listen_start(self):
        LOG.info("Starting up the RealSense pipeline")
        self.pipe = rs.pipeline()
        # Build config object and request pose data
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.pose)

        LOG.info("Connecting to the RealSense")
        # Start streaming with requested config
        self.pipe.start(self.cfg)
        LOG.info("The RealSense pipe started")


    def listen_end(self):
        LOG.info("Disconnecting from the RealSense.")
        self.pipe.stop()
        LOG.info("The RealSense pipe stopped.")


    def get_sensory_observation(self):
        """
        acceleration
        angular_acceleration
        angular_velocity
        rotation
        translation
        velocity
        """
        # Wait for the next set of frames from the camera
        frames = self.pipe.wait_for_frames()
        # Fetch pose frame
        pose = frames.get_pose_frame()
        if pose:
            # Print some of the pose data to the terminal
            _data = pose.get_pose_data()
            self.data = [
                _data.translation.x,       # 0
                _data.translation.y,
                _data.translation.z,
                _data.rotation.x,          # 3
                _data.rotation.y,
                _data.rotation.z,
                _data.rotation.w,
                _data.velocity.x,          # 7
                _data.velocity.y,
                _data.velocity.z,
                _data.angular_velocity.x,  # 10
                _data.angular_velocity.y,
                _data.angular_velocity.z,
            ]

        return np.asarray(self.data)


class SumTemporalDerivativeFilter:
    """
    Let's have variable x samples with sample time dt
    x[0], x[1], x[2], x[3], x[4] ...
    then the numeric derivative (slope) is
    dx[i] = (x[i]-x[i-1]) / dt

    Now if we have derivatives dx[1] ... dx[N] and initial state x[0] we can backtrack x[N]
    x[N] = x[0] + dt * SUM_1^N dx[i]
    x[N] = x[0] + {(x[1] - x[0]) + (x[2] - x[1]) ... (x[N]-x[N-1])}/dt
    x[N] = x[0] + {-x[0] + x[N]} = x[N]

    However, we don't always care about derivatives that happened in the past, rather we
    want some temporally-local information about the variable x evolution.
    Let x estimate made from L last time steps be:

    X[i] = dt * SUM_(i-L)^i dx[i]
    X[i] = dt * {(x[i-L] - x[i-L-1])/dt + (x[i-L+1] - x[i-L])/dt + ... + (x[i] - x[i-1])/dt}
    X[i] = dt * {-x[i-L-1] + x[i]} / dt
    X[i] = x[i] - x[i-L-1]

    """

    def __init__(self, sample_length: int, dimension: int):
        self.buffer = np.zeros((sample_length, dimension))
        self.idx = 0
        self.length = sample_length

    def __call__(self, value: np.ndarray):
        self.buffer[self.idx, :] = value
        next_idx = (self.idx + 1) % self.length
        self.idx = next_idx
        return value - self.buffer[next_idx, :]

def get_heading(q):
    # arcsin(2XY + 2ZW)
    roll = np.arcsin(2 * q[0] * q[1] + 2 * q[2] * q[3])

    # arctan2(2XW-2YZ, 1-2XX-2ZZ)
    pitch = np.arctan2(2 * q[0] * q[3] - 2 * q[1] * q[2], 1 - 2 * q[0] * q[0] - 2 * q[2] * q[2])

    # arctan2(2YW-2XZ, 1-2YY-2ZZ)
    yaw = np.arctan2(2 * q[1] * q[3] - 2 * q[0] * q[2], 1 - 2 * q[1] * q[1] - 2 * q[2] * q[2])

    return np.asarray([roll, pitch, yaw])


def transform_axes(xyz_realsense):
    return xyz_realsense.dot(REALSENSE_ROBOT_TRANSFER)


def position_and_heading(sensory_observation):
    position = transform_axes(sensory_observation[ID_TRANSLATION:ID_TRANSLATION+3])
    heading = get_heading(sensory_observation[ID_ROTATION:ID_ROTATION+4])
    return position, heading


def heading_speed(roll_pitch_yaw, velocity_vector):
    vel_dir = np.arctan2(velocity_vector[1], velocity_vector[0])
    vel_amp = np.sqrt(np.sum(np.square(velocity_vector[:2])))
    return vel_amp * np.cos(vel_dir - roll_pitch_yaw[2])


def heading_and_side_speeds(roll_pitch_yaw, velocity_vector):
    vel_dir = np.arctan2(velocity_vector[1], velocity_vector[0])
    vel_amp = np.sqrt(np.sum(np.square(velocity_vector[:2])))
    return [vel_amp * np.cos(vel_dir - roll_pitch_yaw[2]), vel_amp * np.cos(vel_dir - roll_pitch_yaw[2] + _RIGHT_ANGLE)]


def extract_heading_speed(sensory_observation):
    velocity_vector = transform_axes(sensory_observation[ID_VELOCITY:ID_VELOCITY+3])
    rotation = get_heading(sensory_observation[ID_ROTATION:ID_ROTATION+4])
    return heading_speed(rotation, velocity_vector)


def extract_heading_and_side_speeds(sensory_observation):
    velocity_vector = transform_axes(sensory_observation[ID_VELOCITY:ID_VELOCITY+3])
    rotation = get_heading(sensory_observation[ID_ROTATION:ID_ROTATION+4])
    return heading_and_side_speeds(rotation, velocity_vector)


def extract_angular_velocity(sensory_observation):
    return transform_axes(sensory_observation[ID_ANGULAR_VELOCITY:ID_ANGULAR_VELOCITY+3])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import coupling_evol.data_process.inprocess.records as R

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
    # R = None


    rsl = RealSenseListener()
    rsl.listen_start()
    _ys = []
    for i in range(1000):
        _ys.append(rsl.get_sensory_observation())
        print(_ys[-1])
        time.sleep(0.01)
    rsl.listen_end()

    _ys = np.asarray(_ys)

    rec = {
        "pos": _ys[:, :3],
        "head": _ys[:, 3:7],
        "vel_lin": _ys[:, 7:10],
        "vel_ang": _ys[:, 10:13],
    }

    R.save_records("data.hdf5", rec)
    R.print_record_shapes(rec)

    r = R.load_records("data.hdf5")[0]
    # ##
    # plt.title("X-Y")
    # plt.scatter(r["pos"][:, 0], r["pos"][:, 1], label="pos")
    # plt.legend()
    # ##
    # plt.figure()
    # plt.title("X-Z")
    # plt.scatter(r["pos"][:, 0], r["pos"][:, 2], label="pos")
    # ##

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(r["pos"][:, 0], r["pos"][:, 1], r["pos"][:, 2])

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(r["vel_lin"][:, 0], r["vel_lin"][:, 1], r["vel_lin"][:, 2])

    ##
    # fig = plt.figure()
    # _f = fig.subplots(3, 3)
    # for j, seg in enumerate(["pos", "vel_lin", "vel_ang"]):
    #     f = _f[j]
    #     for i, dim in enumerate(["x", "y", "z"]):
    #         f[i].plot(r[seg][:, i])
    #         if j == 1:
    #             f[i].set_xlabel(dim)
    #     f[0].set_ylabel(seg)

    ##
    fig = plt.figure()
    _f = fig.subplots(3, 3)
    for j, seg in enumerate(["pos", "vel_lin", "vel_ang"]):
        f = _f[j]
        seg_t = np.asarray([transform_axes(r[seg][k, :]) for k in range(r[seg].shape[0])])
        for i, dim in enumerate(["x", "y", "z"]):
            f[i].plot(seg_t[:, i])
            if j == 1:
                f[i].set_xlabel(dim)
        f[0].set_ylabel(seg)



    fig = plt.figure()
    _f = fig.subplots(3, 4)
    h = np.asarray([get_heading(r["head"][t, :]) for t in range(r["head"].shape[0])])

    f = _f[0]
    h = np.asarray(h)
    for i, dim in enumerate(["x", "y", "z","w"]):
        f[i].plot(r["head"][:, i])

    f = _f[1]
    h = np.asarray(h)
    for i, dim in enumerate(["x-roll", "y-pitch", "z-yaw"]):
        f[i].plot(h[:, i])
        f[i].set_xlabel(dim)

    f = _f[2]
    h = np.asarray(h)
    for i, dim in enumerate(["x-roll", "y-pitch", "z-yaw"]):
        f[i].plot(r["vel_ang"][:, i])

    # velocities = np.asarray([transform_axes(r["vel_lin"][k, :]) for k in range(r["vel_lin"].shape[0])])
    # vel_dir = np.asarray([np.arctan2(velocities[k, 1], velocities[k, 0]) for k in range(r["vel_lin"].shape[0])])
    # vel_amp = np.sqrt(np.sum(np.square(velocities[:, :2]), axis=1))
    # a = vel_amp * np.cos(vel_dir - h[:, 2])
    #     "pos": _ys[:, :3],
    #     "head": _ys[:, 3:7],
    #     "vel_lin": _ys[:, 7:10],
    #     "vel_ang": _ys[:, 10:13],
    raws = [
        np.concatenate([r["pos"][t, :], r["head"][t, :], r["vel_lin"][t, :], r["vel_ang"][t, :]])
        for t in range(r["vel_lin"].shape[0])
    ]

    y_out = [
        np.concatenate([[extract_heading_speed(raw)], extract_angular_velocity(raw)]) for raw in raws
        ]
    y_out = np.asarray(y_out)

    fig = plt.figure()
    fig.suptitle("y_out")
    _f = fig.subplots(4, 1)
    for i, seg in enumerate(["head_speed", "roll_speed", "pitch_speed", "yaw_speed"]):
        _f[i].plot(y_out[:, i])
        _f[i].set_ylabel(seg)


    plt.show()