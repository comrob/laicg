import numpy
from typing import List, Dict
import numpy as np

from coupling_evol.data_process.inprocess.record_logger import RECORDER_T
from coupling_evol.engine.common import LOGGER_T, RecordType


class EmbeddedTargetParameter(object):
    VALUE_NAME = ""
    WEIGHT_NAME = "_weight"
    METRIC_NAME = "_metric"

    def __init__(self, value: numpy.ndarray, weight: numpy.ndarray, metric: numpy.ndarray):
        """
        Sensory target description.
        Difference (x - target)
        @param value: (sensory dim, granularity) Target value.
        @param weight: (sensory dim, granularity) Weight of the difference between target and measurement.
        @param metric: (sensory dim, granularity)
        1: only positive (more than target) difference, -1: only negative (less than target) difference, 0: both
        """
        self.value = value
        self.weight = weight
        self.metric = metric
        self.metric_neg = self.metric <= 0.
        self.metric_pos = self.metric >= 0.
        self.shape = value.shape

    @staticmethod
    def simple_target(value: numpy.ndarray):
        return EmbeddedTargetParameter(value, weight=np.ones(value.shape), metric=np.zeros(value.shape))

    def embedding_difference(self, sensory_embedding: numpy.ndarray):
        weighted = (sensory_embedding - self.value) * self.weight
        return np.maximum(weighted, 0) * self.metric_pos + np.minimum(weighted, 0) * self.metric_neg

    @staticmethod
    def difference_mem(target_mem, metric_mem, weight_mem, sensory_mem):
        weighted = (target_mem - sensory_mem) * weight_mem
        metric_pos = metric_mem >= 0.
        metric_neg = metric_mem <= 0.
        return np.maximum(weighted, 0) * metric_pos + np.minimum(weighted, 0) * metric_neg

    def embedding_phase_sum_difference(self, sensory_embedding: np.ndarray):
        """
        In this mode, the difference is parametrized by FIRST phase target params!
        @param sensory_embedding:
        @type sensory_embedding:
        @return:
        @rtype:
        """
        sensor_agg = np.sum(sensory_embedding, axis=1)
        diff = (sensor_agg - self.value[:, 0]) * self.weight[:, 0]
        return np.maximum(diff, 0) * self.metric_pos[:, 0] + np.minimum(diff, 0) * self.metric_neg[:, 0]

    def get_tuple(self):
        return self.value, self.weight, self.metric

    def write_into_dictionary(self, d: dict, prefix):
        d[prefix + self.VALUE_NAME] = np.zeros(self.value.shape) + self.value
        d[prefix + self.WEIGHT_NAME] = np.zeros(self.weight.shape) + self.weight
        d[prefix + self.METRIC_NAME] = np.zeros(self.metric.shape) + self.metric

    def write_into_logger(self, lgg: RECORDER_T, prefix):
        lgg(prefix + self.VALUE_NAME, np.zeros(self.value.shape) + self.value)
        lgg(prefix + self.WEIGHT_NAME, np.zeros(self.weight.shape) + self.weight)
        lgg(prefix + self.METRIC_NAME, np.zeros(self.metric.shape) + self.metric)

    @classmethod
    def read_from_record(cls, rec: dict, prefix, idx=None):
        if idx is None:
            return cls(rec[prefix + cls.VALUE_NAME], rec[prefix + cls.WEIGHT_NAME],
                       rec[prefix + cls.METRIC_NAME])
        else:
            return cls(rec[prefix + cls.VALUE_NAME][idx], rec[prefix + cls.WEIGHT_NAME][idx],
                       rec[prefix + cls.METRIC_NAME][idx])

    @classmethod
    def read_numpy_from_record(cls, rec: dict, prefix):
        return rec[prefix + cls.VALUE_NAME], rec[prefix + cls.WEIGHT_NAME], rec[prefix + cls.METRIC_NAME]

    @classmethod
    def list_from_record(cls, rec: dict, prefix) -> List:
        vals, ws, ms = cls.read_numpy_from_record(rec, prefix)
        n = vals.shape[0]
        return [cls(vals[i, :, :], weight=ws[i, :, :], metric=ms[i, :, :]) for i in range(n)]

    @staticmethod
    def target_diff_subsample(record, target_key, sensory_signal_key, phase_key):
        raw_trg_signal = record[target_key]
        sens_signal = record[sensory_signal_key]
        phase_signal = record[phase_key]
        phase_nums = np.argmax(phase_signal, axis=1)
        phase_is_valid = np.sum(phase_signal, axis=1) == 1
        n, sens_dim, ph_dim = raw_trg_signal.shape
        embedding_buffer = np.zeros((sens_dim, ph_dim))
        current_phase = 0
        signal_buffer = []
        current_diff = np.zeros((sens_dim, ph_dim))
        current_sum_diff = np.zeros((sens_dim, ))
        ret = []
        ret_sum = []
        for i in range(n):
            if phase_is_valid[i]:
                if current_phase != phase_nums[i] and len(signal_buffer) > 0:
                    embedding_buffer[:, current_phase] = np.mean(signal_buffer, axis=0)
                    current_phase = phase_nums[i]
                    signal_buffer = []
                    ##
                    curr_trg = EmbeddedTargetParameter.read_from_record(record, target_key, i)
                    current_diff = curr_trg.embedding_difference(embedding_buffer)
                    current_sum_diff = curr_trg.embedding_phase_sum_difference(embedding_buffer)
                signal_buffer.append(sens_signal[i])
            ret_sum.append(current_sum_diff)
            ret.append(current_diff)
        return np.asarray(ret), np.asarray(ret_sum)


class EmbeddingController(object):
    DELTA_GAIT = "delta_u"
    CONTROL_PSFX = "ctr"

    """
    Represents fast paced controller that provides the embeddings "immediately" for given target and sensory.
    Here should be not "offline" learning, i.e., each call should have about the same duration.
    """

    def get_history(self) -> List[Dict[str, np.ndarray]]:
        pass

    def __call__(self, sensory_embedding: numpy.ndarray, target_parameter: EmbeddedTargetParameter,
                 motion_phase: numpy.ndarray) -> numpy.ndarray:
        """

        @param sensory_input: current sensory measurement (sensor dim, granularity)
        @param target_parameter: defines the target
        @param motion_phase: (granularity, )
        @return motor command embedding (motor dim, granularity)
        """
        pass


class ConstantEmbeddingController(EmbeddingController):
    GAIT = "gait"

    def __init__(self, gait):
        self.gait = gait

    def get_history(self) -> List[Dict[str, np.ndarray]]:
        return [{self.GAIT: self.gait}]

    def __call__(self, *args, **kwargs):
        return self.gait


if __name__ == '__main__':
    trg_at_one = EmbeddedTargetParameter(value=np.ones((1, 3)), weight=np.ones((1, 3)), metric=np.zeros((1, 3)))
    trg_below_one = EmbeddedTargetParameter(value=np.ones((1, 3)), weight=np.ones((1, 3)), metric=np.ones((1, 3)))
    trg_above_one = EmbeddedTargetParameter(value=np.ones((1, 3)), weight=np.ones((1, 3)), metric=-np.ones((1, 3)))

    obs = np.zeros((1, 3))
    print(f"at_one(zero) = {trg_at_one.embedding_difference(obs)} = -1")
    print(f"above_one(zero) = {trg_above_one.embedding_difference(obs)} = -1")
    print(f"below_one(zero) = {trg_below_one.embedding_difference(obs)} = 0")

    obs = np.ones((1, 3))
    print(f"at_one(one) = {trg_at_one.embedding_difference(obs)} = 0")
    print(f"above_one(one) = {trg_above_one.embedding_difference(obs)} = 0")
    print(f"below_one(one) = {trg_below_one.embedding_difference(obs)} = 0")

    obs = np.ones((1, 3)) * 2
    print(f"at_one(two) = {trg_at_one.embedding_difference(obs)} = 1")
    print(f"above_one(two) = {trg_above_one.embedding_difference(obs)} = 0")
    print(f"below_one(two) = {trg_below_one.embedding_difference(obs)} = 1")


    obs = np.asarray([[-1, 0, 1]])
    print(f"SUM at_one(SUM zero) = {trg_at_one.embedding_phase_sum_difference(obs)} = -1")
    print(f"SUM above_one(SUM zero) = {trg_above_one.embedding_phase_sum_difference(obs)} = -1")
    print(f"SUM below_one(SUM zero) = {trg_below_one.embedding_phase_sum_difference(obs)} = 0")

    obs = np.asarray([[0, 0, 1]])
    print(f"SUM at_one(SUM one) = {trg_at_one.embedding_phase_sum_difference(obs)} = 0")
    print(f"SUM above_one(SUM one) = {trg_above_one.embedding_phase_sum_difference(obs)} = 0")
    print(f"SUM below_one(SUM one) = {trg_below_one.embedding_phase_sum_difference(obs)} = 0")

    obs = np.asarray([[1, 1, 0]])
    print(f"SUM at_one(SUM one) = {trg_at_one.embedding_phase_sum_difference(obs)} = 1")
    print(f"SUM above_one(SUM one) = {trg_above_one.embedding_phase_sum_difference(obs)} = 0")
    print(f"SUM below_one(SUM one) = {trg_below_one.embedding_phase_sum_difference(obs)} = 1")