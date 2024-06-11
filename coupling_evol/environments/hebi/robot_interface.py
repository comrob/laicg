import os.path

import hebi
from time import sleep
import numpy as np
import logging
import os
LOG = logging.getLogger(__name__)

LEG_PSFXS = ['J1_base', 'J2_shoulder', 'J3_elbow']


def get_symmetric_joint_command(odd_leg_command):
    evn_leg_command = odd_leg_command * np.asarray([1, -1, -1])
    joint_command = np.zeros(18)

    joint_command[0:3] = odd_leg_command
    joint_command[3:6] = evn_leg_command
    joint_command[6:9] = odd_leg_command
    joint_command[9:12] = evn_leg_command
    joint_command[12:15] = odd_leg_command
    joint_command[15:18] = evn_leg_command
    return joint_command


class CommandAndFeedback:
    """
    Wrapper through which the commands are sent and feedback received.
    """
    def __init__(self, group, feedback_frequency=100., gains_path=None):
        self.group = group
        self.group_command = hebi.GroupCommand(group.size)
        self.group_feedback = hebi.GroupFeedback(group.size)
        self.group.feedback_frequency = feedback_frequency
        self.group.command_lifetime = 100.0

        if gains_path is not None:
            LOG.info(f"Loading gains from {gains_path}")
            configuration_command = hebi.GroupCommand(group.size)
            configuration_command.read_gains(gains_path)
            LOG.info(f"Sending servo configuration.")
            self.group.send_command_with_acknowledgement(configuration_command)
            sleep(1)


    def send_command(self, positions, efforts):
        self.group_command.position = positions
        self.group_command.effort = efforts
        self.group.send_command(self.group_command)

    def update_feedback(self):
        fbk = self.group_feedback
        group_feedback = self.group.get_next_feedback(reuse_fbk=self.group_feedback)
        if group_feedback is None:
            self.group_feedback = fbk

    def get_position_feedback(self):
        return self.group_feedback.position

    def get_effort_feedback(self):
        return self.group_feedback.effort


class Lily(object):
    """
    This class will contain configurations and some motion elements relevant to the Lily robot.
    """


    # L1_J1, L1_J2, L1_J3 ..., L6_J3
    GROUP_NAMES = [f"L{i + 1}_" + psf for i in range(6) for psf in LEG_PSFXS]

    SHOULDER_JOINTS = [1, 4, 7, 10, 13, 16]
    BASE_JOINTS = [0, 3, 6, 9, 12, 15]

    #                       J1,  J2,   J3
    RESTING_LEG_POSITION = [0, -0.8, -1.75]  # radians
    STANDING_LEG_POSITION = [0, 0.2, -1.20]  # radians

    STANDING_LEG_POSITION_INTERVAL = [0.2, .4, 0.2]

    ###
    JOINT_STIFFNESS = np.asarray([16, 40, 8] * 6)
    STANDING_JOINTS_POS = get_symmetric_joint_command(STANDING_LEG_POSITION)
    STANDING_JOINTS_INTERVAL = np.abs(get_symmetric_joint_command(STANDING_LEG_POSITION_INTERVAL))

    #
    # GAINS_PATH = "gains18.xml"
    # GAINS_PATH = "saved_gains.xml"
    GAINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_gains2.xml")
    def __init__(self):
        lookup = hebi.Lookup()
        sleep(2)
        LOG.info('Modules found on network:')
        for entry in lookup.entrylist:
            LOG.info(f'{entry.family} | {entry.name}')

        self.group = lookup.get_group_from_names(['Lily'], self.GROUP_NAMES)
        ##

    def init_command_and_feedback(self):
        return CommandAndFeedback(self.group, gains_path=self.GAINS_PATH)
        # return CommandAndFeedback(self.group)#, gains_path=Lily.GAINS_PATH)#, safety_path=Lily.GAINS_PATH)

    @classmethod
    def position_change_by_effort(clz, caf: CommandAndFeedback, position_start, position_end, stiffness, cycles_to_life=1000,
                                  alpha_end=0.8, shoulders_are_positional=True, shoulder_pos_eps=0.2):
        alpha_end_iter = int(cycles_to_life * alpha_end)
        for i in range(cycles_to_life):
            ## getting the current position
            caf.update_feedback()
            pos = caf.get_position_feedback()
            ## early end condition
            diff = (position_end - pos)
            shoulder_diff = np.abs(diff[clz.SHOULDER_JOINTS])
            if np.alltrue(shoulder_diff < shoulder_pos_eps):
                LOG.debug("All shoulder positions within interval. Ending position change.")
                return
            ## calc new reference position
            alpha = np.minimum(i / alpha_end_iter, 1)
            ref_in = position_start * (1 - alpha) + position_end * alpha
            ## effort control
            effort = (ref_in - pos) * stiffness
            position = np.zeros(ref_in.shape) + np.nan

            if shoulders_are_positional:
                effort[clz.SHOULDER_JOINTS] = np.nan
                position[clz.SHOULDER_JOINTS] = ref_in[clz.SHOULDER_JOINTS]

            caf.send_command(position, effort)
            # logger.debug(f'{i}/{cycles_to_life}, Position Ref/Fb:{position_end}/{pos}')
            sleep(1.0 / caf.group.feedback_frequency)

    @classmethod
    def stand_up(clz, caf: CommandAndFeedback):
        LOG.info("Standing up process initiated.")
        ref_pos_preparation = get_symmetric_joint_command(clz.RESTING_LEG_POSITION)
        ref_pos_standing = get_symmetric_joint_command(clz.STANDING_LEG_POSITION)
        # stiffness (base, shoulder, elbow)
        stiffness = np.asarray([8, 0, 4] * 6)  # [Nm/rad]

        ## getting the current position
        caf.update_feedback()
        pos = caf.get_position_feedback()
        ## go to rest
        LOG.info(f"Lily setting to rest position. Pos: {pos} to {ref_pos_preparation}")
        clz.position_change_by_effort(caf, pos, ref_pos_preparation, stiffness, cycles_to_life=800, alpha_end=0.8,
                                       shoulders_are_positional=True, shoulder_pos_eps=0.2)
        ## getting the current position
        caf.update_feedback()
        pos = caf.get_position_feedback()
        ## go to stance
        LOG.info(f"Lily setting to standing. Pos :{pos} to {ref_pos_standing}")
        clz.position_change_by_effort(caf, ref_pos_preparation, ref_pos_standing, stiffness, cycles_to_life=900, alpha_end=0.9,
                                       shoulders_are_positional=True, shoulder_pos_eps=0.1)

    @staticmethod
    def safe_fix_position(caf: CommandAndFeedback, position, cycles=200, safety_eps=0.1):
        LOG.info(f"Trying to fix the position {position}.")
        #
        caf.update_feedback()
        pos = caf.get_position_feedback()
        #

        diff = (position - pos)
        shoulder_diff = np.abs(diff)
        if not np.alltrue(shoulder_diff < safety_eps):
            LOG.warning(f"Will not try to fix because current position is too far. Diff: {shoulder_diff} ")
            return False

        effort = np.zeros(position.shape) + np.nan

        for i in range(cycles):
            if i % 100 == 0:
                LOG.info(f"Fixing cycle {i}/{cycles}")
            caf.send_command(position, effort)
            sleep(1 / caf.group.feedback_frequency)
        return True

    @classmethod
    def lie_down(clz, caf: CommandAndFeedback):
        LOG.info("Lying down process initiated.")
        ref_pos_rest = get_symmetric_joint_command(clz.RESTING_LEG_POSITION)
        # stiffness (base, shoulder, elbow)
        stiffness = np.asarray([8, 0, 4] * 6)  # [Nm/rad]

        ## getting the current position
        caf.update_feedback()
        pos = caf.get_position_feedback()
        ## go to rest
        clz.position_change_by_effort(caf, pos, ref_pos_rest, stiffness, cycles_to_life=700, alpha_end=0.99,
                                       shoulders_are_positional=True, shoulder_pos_eps=0.1)

    @classmethod
    def get_position_boundaries(cls):
        """

        @return: rest postion, and max absolute
        """
        return get_symmetric_joint_command(cls.STANDING_LEG_POSITION),\
               np.abs(get_symmetric_joint_command(cls.STANDING_LEG_POSITION_INTERVAL))

    @classmethod
    def de_norm(cls, pos_nrm):
        return cls.STANDING_JOINTS_POS + pos_nrm * cls.STANDING_JOINTS_INTERVAL


class Daisy(object):
    """
    This class will contain configurations and some motion elements relevant to the Lily robot.
    """


    # L1_J1, L1_J2, L1_J3 ..., L6_J3
    GROUP_NAMES = [f"L{i + 1}_" + psf for i in range(6) for psf in LEG_PSFXS]

    SHOULDER_JOINTS = [1, 4, 7, 10, 13, 16]
    BASE_JOINTS = [0, 3, 6, 9, 12, 15]

    #                       J1,  J2,   J3
    RESTING_LEG_POSITION = [0, -0.8, -1.75]  # radians
    STANDING_LEG_POSITION = [0, 0.37, -1.30]  # radians

    # STANDING_LEG_POSITION_INTERVAL = [1., 2., 1.]
    # STANDING_LEG_POSITION_INTERVAL = [0.5, 1., 0.5]
    # STANDING_LEG_POSITION_INTERVAL = [0.25, .5, 0.25]
    STANDING_LEG_POSITION_INTERVAL = [0.2, .4, 0.2]

    ###
    JOINT_STIFFNESS = np.asarray([16, 50, 50] * 6)
    STANDING_JOINTS_POS = get_symmetric_joint_command(STANDING_LEG_POSITION)
    STANDING_JOINTS_INTERVAL = np.abs(get_symmetric_joint_command(STANDING_LEG_POSITION_INTERVAL))

    GAINS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_gains2.xml")

    def __init__(self):
        lookup = hebi.Lookup()
        sleep(2)
        LOG.info('Modules found on network:')
        for entry in lookup.entrylist:
            LOG.info(f'{entry.family} | {entry.name}')

        self.group = lookup.get_group_from_names(['Daisy'], self.GROUP_NAMES)
        ##

    def init_command_and_feedback(self):
        return CommandAndFeedback(self.group, gains_path=Lily.GAINS_PATH)
        # return CommandAndFeedback(self.group)#, gains_path=Lily.GAINS_PATH)#, safety_path=Lily.GAINS_PATH)

    @classmethod
    def position_change_by_position(cls, caf: CommandAndFeedback, position_start, position_end, cycles_to_life=1000,
                                  alpha_end=0.8):
        alpha_end_iter = int(cycles_to_life * alpha_end)
        for i in range(cycles_to_life):
            ## getting the current position
            caf.update_feedback()
            ## calc new reference position
            alpha = np.minimum(i / alpha_end_iter, 1)
            ref_in = position_start * (1 - alpha) + position_end * alpha
            position = ref_in
            effort = np.zeros(ref_in.shape) + np.nan
            caf.send_command(position, effort)
            sleep(1.0 / caf.group.feedback_frequency)

    @classmethod
    def position_change_by_effort(cls, caf: CommandAndFeedback, position_start, position_end, stiffness, cycles_to_life=1000,
                                  alpha_end=0.8, shoulders_are_positional=True, shoulder_pos_eps=0.2):
        alpha_end_iter = int(cycles_to_life * alpha_end)
        for i in range(cycles_to_life):
            ## getting the current position
            caf.update_feedback()
            pos = caf.get_position_feedback()
            ## early end condition
            diff = (position_end - pos)
            shoulder_diff = np.abs(diff[cls.SHOULDER_JOINTS])
            if np.alltrue(shoulder_diff < shoulder_pos_eps):
                LOG.debug("All shoulder positions within interval. Ending position change.")
                return
            ## calc new reference position
            alpha = np.minimum(i / alpha_end_iter, 1)
            ref_in = position_start * (1 - alpha) + position_end * alpha
            ## effort control
            effort = (ref_in - pos) * stiffness
            position = np.zeros(ref_in.shape) + np.nan

            if shoulders_are_positional:
                effort[cls.SHOULDER_JOINTS] = np.nan
                position[cls.SHOULDER_JOINTS] = ref_in[cls.SHOULDER_JOINTS]

            caf.send_command(position, effort)
            # logger.debug(f'{i}/{cycles_to_life}, Position Ref/Fb:{position_end}/{pos}')
            sleep(1.0 / caf.group.feedback_frequency)

    @classmethod
    def stand_up(clz, caf: CommandAndFeedback):
        LOG.info("Standing up process initiated.")
        ref_pos_preparation = get_symmetric_joint_command(clz.RESTING_LEG_POSITION)
        ref_pos_standing = get_symmetric_joint_command(clz.STANDING_LEG_POSITION)
        # stiffness (base, shoulder, elbow)

        ## getting the current position
        caf.update_feedback()
        pos = caf.get_position_feedback()
        ## go to rest
        LOG.info(f"Daisy setting to rest position. Pos: {pos} to {ref_pos_preparation}")
        clz.position_change_by_position(caf, pos, ref_pos_preparation, cycles_to_life=600, alpha_end=0.8)
        ## getting the current position
        caf.update_feedback()
        pos = caf.get_position_feedback()
        ## go to stance
        LOG.info(f"Daisy setting to standing. Pos :{pos} to {ref_pos_standing}")
        clz.position_change_by_position(caf, ref_pos_preparation, ref_pos_standing, cycles_to_life=600, alpha_end=0.9)

    @staticmethod
    def safe_fix_position(caf: CommandAndFeedback, position, cycles=200, safety_eps=0.1):
        LOG.info(f"Trying to fix the position {position}.")
        #
        caf.update_feedback()
        pos = caf.get_position_feedback()
        #

        diff = (position - pos)
        shoulder_diff = np.abs(diff)
        if not np.alltrue(shoulder_diff < safety_eps):
            LOG.warning(f"Will not try to fix because current position is too far. Diff: {shoulder_diff} ")
            return False

        effort = np.zeros(position.shape) + np.nan

        for i in range(cycles):
            if i % 100 == 0:
                LOG.info(f"Fixing cycle {i}/{cycles}")
            caf.send_command(position, effort)
            sleep(1 / caf.group.feedback_frequency)
        return True

    @classmethod
    def lie_down(cls, caf: CommandAndFeedback):
        LOG.info("Lying down process initiated.")
        ref_pos_rest = get_symmetric_joint_command(cls.RESTING_LEG_POSITION)
        ## getting the current position
        caf.update_feedback()
        pos = caf.get_position_feedback()
        ## go to rest
        cls.position_change_by_position(caf, pos, ref_pos_rest, cycles_to_life=600, alpha_end=0.99)

    @classmethod
    def get_position_boundaries(cls):
        """

        @return: rest postion, and max absolute
        """
        return get_symmetric_joint_command(cls.STANDING_LEG_POSITION),\
               np.abs(get_symmetric_joint_command(cls.STANDING_LEG_POSITION_INTERVAL))

    @classmethod
    def de_norm(cls, pos_nrm):
        return cls.STANDING_JOINTS_POS + pos_nrm * cls.STANDING_JOINTS_INTERVAL


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
    # claz = Lily
    claz = Daisy
    robot = claz()
    _caf = robot.init_command_and_feedback()
    claz.stand_up(_caf)
    claz.safe_fix_position(_caf, get_symmetric_joint_command(claz.STANDING_LEG_POSITION), safety_eps=2, cycles=5000)
    claz.lie_down(_caf)
