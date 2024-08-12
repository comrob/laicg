import numpy as np
from coupling_evol.engine.embedded_control import EmbeddedTargetParameter

import coupling_evol.data_process.inprocess.recorder as rlog
from coupling_evol.engine import common as C
BABBLE_PSFX = ""
REC = rlog.get_recorder(prefix=C.RecordNamespace.LIFE_CYCLE.key, postfix=BABBLE_PSFX)


class MotorBabbler:
    """
    Motor babbling holds noise motor embedding which is updated whenever the phase_activation change occurs. The update
    concerns only the phase after the current. In such a way, the same random babble is measured during the entire
    segment.
    """
    BABBLE_NAME = "nu_bbl"
    ACTIVATION_NAME = "a_bbl"

    def __init__(self, motor_dim, granularity, babble_generator: callable):
        self.motor_dim = motor_dim
        self.granularity = granularity
        self.previous_ph_id = -1
        self.babble = np.zeros((motor_dim, granularity))
        self.babble_generator = babble_generator
        #

    def step(self, sensory_embedding: np.ndarray, target_parameter: EmbeddedTargetParameter,
                 phase_activation: np.ndarray):
        current_ph_id = np.argmax(phase_activation)
        if current_ph_id != self.previous_ph_id:
            next_ph_id = np.mod(current_ph_id + 1, self.granularity)
            self.babble[:, next_ph_id] = self.babble_generator()
        self.previous_ph_id = current_ph_id

        REC(self.BABBLE_NAME, np.zeros(self.babble.shape) + self.babble)
        REC(self.ACTIVATION_NAME, np.zeros(phase_activation.shape) + phase_activation)

    def current_babble(self):
        return self.babble

    def reset(self):
        pass

class MotorBabblerConservative(MotorBabbler):
    """
    Motor babbling holds noise motor embedding which is updated whenever the phase_activation change occurs. The update
    concerns only the phase after the current. In such a way, the same random babble is measured during the entire
    segment.
    """
    BABBLE_NAME = "nu_bbl"
    ACTIVATION_NAME = "a_bbl"

    def __init__(self, motor_dim, granularity, babble_generator: callable,
                 max_random_segments=1, gait_pick_prob=1/2):
        super().__init__(motor_dim, granularity, babble_generator)
        self.motor_dim = motor_dim
        self.granularity = granularity
        self.previous_ph_id = -1
        self.babble = np.zeros((motor_dim, granularity))
        self.babble_generator = babble_generator
        self.max_random_segments = max_random_segments
        self.gait_pick_prob = gait_pick_prob
        #

    def step(self, sensory_embedding: np.ndarray, target_parameter: EmbeddedTargetParameter,
                 phase_activation: np.ndarray):
        current_ph_id = np.argmax(phase_activation)
        if current_ph_id != self.previous_ph_id and current_ph_id == 0:
            babb = np.zeros(self.babble.shape)
            if self.gait_pick_prob > np.random.uniform():
                for i in range(self.max_random_segments):
                    ph = np.random.choice(self.babble.shape[1])
                    babb[:, ph] = self.babble_generator()
            self.babble = babb
        self.previous_ph_id = current_ph_id

        ##
        REC(self.BABBLE_NAME, np.zeros(self.babble.shape) + self.babble)
        REC(self.ACTIVATION_NAME, np.zeros(phase_activation.shape) + phase_activation)

    def current_babble(self):
        return self.babble

class DynamicMotorBabbler(MotorBabbler):
    INIT_REST_GAITS = 2
    NUMERICAL_EPS = 0.000001
    BABBLE_WEIGHTS = "bbl_weight"

    def __init__(self, motor_dim, granularity, babble_generator: callable, learning_rate=0.0001):
        super().__init__(motor_dim, granularity, babble_generator)
        self.motor_dim = motor_dim
        self.granularity = granularity
        self.previous_ph_id = -1
        self.babble = np.zeros((motor_dim, granularity))
        self.babble_generator = babble_generator

        self.valid_sensory_embedding = np.zeros((1, granularity))
        self.fin_phases_count = 0
        self.fin_gaits_count = 0
        self.diff_mem = []
        self.diff_exp = 0

        self.babble_scored_mem = [np.zeros((motor_dim, granularity))+0.0001]
        self.babble_weights = np.zeros((motor_dim, granularity)) + 1/(motor_dim * granularity)
        self.error_mem = [0.1]
        self.babble_mem = [np.zeros((motor_dim, granularity))+0.0001]
        #
        self.learning_rate = learning_rate

    def reset(self):
        super().reset()
        self.previous_ph_id = -1
        self.babble = np.zeros((self.motor_dim, self.granularity))

        self.valid_sensory_embedding = np.zeros((1, self.granularity))
        self.fin_phases_count = 0
        self.fin_gaits_count = 0
        self.diff_mem = []
        self.diff_exp = 0

        self.babble_scored_mem = [np.zeros((self.motor_dim, self.granularity))+0.0001]
        self.babble_weights = np.ones((self.motor_dim, self.granularity))/(self.motor_dim * self.granularity)
        self.error_mem = [0.1]
        self.babble_mem = [np.zeros((self.motor_dim, self.granularity))+0.0001]

    def update_babble_weights(self):
        babble_score = np.mean(self.babble_scored_mem, axis=0)
        babb_std = np.maximum(np.std(self.babble_mem, axis=0), self.NUMERICAL_EPS)
        errdiff_std = np.maximum(np.std(self.error_mem), self.NUMERICAL_EPS)
        sq_corr = np.square(babble_score/(babb_std * errdiff_std))
        self.babble_weights += (sq_corr/np.sum(sq_corr) - self.babble_weights) * self.learning_rate

    def step(self, sensory_embedding: np.ndarray, target_parameter: EmbeddedTargetParameter,
                 phase_activation: np.ndarray):
        current_ph_id = np.argmax(phase_activation)
        if self.previous_ph_id == -1:
            self.valid_sensory_embedding = sensory_embedding
            self.current_ph_id = current_ph_id

        if current_ph_id != self.previous_ph_id:  # new phase
            self.fin_phases_count += 1
            ## dynamics
            self.valid_sensory_embedding[:, self.previous_ph_id] = sensory_embedding[:, self.previous_ph_id]
            if self.fin_phases_count == len(phase_activation):  # new gait
                self.fin_phases_count = 0
                self.fin_gaits_count += 1

                if self.fin_gaits_count <= self.INIT_REST_GAITS:
                    self.diff_mem.append(target_parameter.embedding_difference(self.valid_sensory_embedding))

                if self.fin_gaits_count == self.INIT_REST_GAITS+1:
                    self.diff_exp = np.mean(self.diff_mem, axis=0)
                    self.diff_mem = []

            ## babble
            if self.fin_gaits_count > self.INIT_REST_GAITS:
                prev_trg_diff = target_parameter.embedding_difference(self.valid_sensory_embedding)[:,
                                self.previous_ph_id]
                diff_dot = np.mean(np.square(prev_trg_diff) - np.square(self.diff_exp[:, self.previous_ph_id]))
                self.error_mem.append(diff_dot)
                self.babble_mem.append(np.zeros(self.babble.shape)+self.babble)
                self.babble_scored_mem.append(self.babble * diff_dot)
                self.update_babble_weights()
                ##
                next_ph_id = np.mod(current_ph_id + 1, self.granularity)
                self.babble[:, next_ph_id] = self.babble_generator() * self.babble_weights[:, next_ph_id] * (self.motor_dim * self.granularity)

        self.previous_ph_id = current_ph_id
        REC(self.BABBLE_NAME, np.zeros(self.babble.shape) + self.babble)
        REC(self.ACTIVATION_NAME, np.zeros(phase_activation.shape) + phase_activation)
        REC(self.BABBLE_WEIGHTS, np.zeros(phase_activation.shape) + self.babble_weights)

    def current_babble(self):
        return self.babble
