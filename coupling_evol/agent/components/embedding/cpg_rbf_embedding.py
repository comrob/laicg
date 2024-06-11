import numpy as np
from scipy.special import softmax


import coupling_evol.data_process.inprocess.recorder as rlog
from coupling_evol.engine import common as C
CPG_PSFX = ""
UEMB_B = "u_emb"
REC_CPG = rlog.get_recorder(prefix=C.RecordNamespace.LIFE_CYCLE.key, postfix=CPG_PSFX)
REC_EMB = rlog.get_recorder(prefix=C.RecordNamespace.LIFE_CYCLE.key, postfix=UEMB_B)


class CpgRbf:
    PHASE_NAME = "ph"
    ACTIVATION_NAME = "a"
    ACTIVATION_SOFTMAX_NAME = "a_sfmx"
    PERTURBATION_NAME = "pert"
    NATURAL_FREQUENCY_NAME = "frq_nat"

    def __init__(self, natural_frequency, rbf_epsilon, granularity, step_size):
        # constants
        self.step_size = step_size
        self.granularity = granularity
        # parameters
        self.natural_frequency = natural_frequency
        self.perturbation = 0.
        # variables
        self.phase = 0
        # setting up the RBF network
        output_shape = (self.granularity, )
        self.centers = (2 * np.pi * np.arange(self.granularity) / self.granularity).reshape(output_shape)
        # self.activation_expression = _parameter_rbf(rbf_epsilon, max_norm=True)
        self.activation_expression_soft = _parameter_rbf(rbf_epsilon, softmax_norm=True)
        self.activation_expression_hard = _parameter_rbf(rbf_epsilon, max_norm=True)
        #

    def step(self, perturbation: float, natural_frequency: float):
        self.natural_frequency = natural_frequency
        self.perturbation = perturbation

        d_ph = self.natural_frequency - np.sin(self.phase) * self.perturbation
        self.phase += d_ph * self.step_size
        # RECORD
        REC_CPG(CpgRbf.PHASE_NAME, self.phase)
        REC_CPG(CpgRbf.NATURAL_FREQUENCY_NAME, self.natural_frequency)
        REC_CPG(CpgRbf.PERTURBATION_NAME, self.perturbation)
        REC_CPG(CpgRbf.ACTIVATION_NAME, self.current_activation())
        REC_CPG(CpgRbf.ACTIVATION_SOFTMAX_NAME, self.current_activation_soft())

    def current_activation(self):
        return self.activation_expression_hard(self.phase, self.centers)

    def current_activation_soft(self):
        return self.activation_expression_soft(self.phase, self.centers)


class CpgRbfDiscrete(CpgRbf):
    def __init__(self, natural_frequency, rbf_epsilon, granularity, step_size):
        super().__init__(natural_frequency, rbf_epsilon, granularity, step_size)
        """
        phi = omg * dt * iter
        """
        self._iter_period = int(2 * np.pi / (step_size * natural_frequency))
        self._phase_range = np.linspace(0, 2*np.pi, self._iter_period + 1)[:self._iter_period]
        self._phase_count = 0

    def step(self, perturbation: float, natural_frequency: float):
        self.natural_frequency = natural_frequency
        self.perturbation = perturbation

        d_ph = 1 - int(np.sin(self.phase) * self.perturbation) # FIXME the sync prly rewrite
        self._phase_count += d_ph
        self.phase = self._phase_range[self._phase_count % self._iter_period]
        # RECORD
        REC_CPG(CpgRbf.PHASE_NAME, self.phase)
        REC_CPG(CpgRbf.NATURAL_FREQUENCY_NAME, self.natural_frequency)
        REC_CPG(CpgRbf.PERTURBATION_NAME, self.perturbation)
        REC_CPG(CpgRbf.ACTIVATION_NAME, self.current_activation())
        REC_CPG(CpgRbf.ACTIVATION_SOFTMAX_NAME, self.current_activation_soft())


class Embedder:
    EMBEDDING_NAME = "nu"
    PHASE_ACTIVAITON_NAME = "nu_ph"
    SIGNAL_NAME = "nu_in"

    def __init__(self, dimension: int, granularity: int, combiner: callable):
        """

        @param dimension:
        @param granularity:
        @param combiner: callable that takes current state and next state and somehow combines them
        """
        self.dimension = dimension
        self.granularity = granularity
        self.combiner = combiner
        ##
        self.prev_ph = -1
        self.ph_buffer = []
        ##
        self.embedding = np.zeros((dimension, granularity))
        ## quality control
        self._prev_embedding = np.zeros((dimension, granularity))
        #

    def step(self, phase_activation, signal):
        ph = np.argmax(phase_activation)
        if self.prev_ph != ph:
            self.ph_buffer = [self.embedding[:, ph]]
        self.prev_ph = ph
        self.embedding[:, ph] = self.combiner(self.ph_buffer, signal)
        self.ph_buffer.append(signal)
        # record
        REC_EMB(Embedder.PHASE_ACTIVAITON_NAME, phase_activation)
        REC_EMB(Embedder.EMBEDDING_NAME, np.zeros(self.embedding.shape) + self.embedding)
        REC_EMB(Embedder.SIGNAL_NAME, signal)

    def current_embedding(self):
        return self.embedding


def de_embedding(phase_activation, embedding):
    return embedding[:, np.argmax(phase_activation)]


def soft_de_embedding(phase_activation, embedding):
    return embedding.dot(phase_activation).T


def affine_combiner(old_state_affinity=0.3):
    def combiner(old, new):
        if len(old) == 0:
            return new
        return old[-1] * old_state_affinity + new * (1 - old_state_affinity)
    return combiner


def mean_combiner():
    def combiner(old, new):
        return np.mean(np.asarray(old + [new]), axis=0)
    return combiner


def _parameter_rbf(epsilon, max_norm=False, softmax_norm=False):
    def rbf(phase, center):
        dist = np.abs(np.exp(-1j*phase)-np.exp(-1j*center))
        return np.exp(-np.square(epsilon * dist))

    if max_norm:
        def rbf_max(phase, center):
            rbf_signal = rbf(phase, center)
            ret = np.zeros(rbf_signal.shape)
            ret[np.argmax(rbf_signal)] = 1
            return ret
        return rbf_max
    elif softmax_norm:
        def rbf_softmax(phase, center):
            rbf_signal = rbf(phase, center)
            return softmax(rbf_signal*5)
        return rbf_softmax

    return rbf


