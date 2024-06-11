import numpy as np
_RIGHT_ANGLE = np.pi/2
_ROUND_ANGLE = np.pi * 2

def get_heading(q):
    # arcsin(2XY + 2ZW)
    roll = np.arcsin(2 * q[0] * q[1] + 2 * q[2] * q[3])

    # arctan2(2XW-2YZ, 1-2XX-2ZZ)
    pitch = np.arctan2(2 * q[0] * q[3] - 2 * q[1] * q[2], 1 - 2 * q[0] * q[0] - 2 * q[2] * q[2])

    # arctan2(2YW-2XZ, 1-2YY-2ZZ)
    yaw = np.arctan2(2 * q[1] * q[3] - 2 * q[0] * q[2], 1 - 2 * q[1] * q[1] - 2 * q[2] * q[2])

    return np.asarray([roll, pitch, yaw])


def heading_and_side_speeds(roll_pitch_yaw, velocity_vector):
    vel_dir = np.arctan2(velocity_vector[1], velocity_vector[0])
    vel_amp = np.sqrt(np.sum(np.square(velocity_vector[:2])))
    return [vel_amp * np.cos(vel_dir - roll_pitch_yaw[2]), vel_amp * np.cos(vel_dir - roll_pitch_yaw[2] + _RIGHT_ANGLE)]
