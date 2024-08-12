from coupling_evol.agent.lifecycle.compound_model import CompoundModel, one_hot_matrix
from coupling_evol.agent.components.ensemble_dynamics.common import EnsembleDynamics
from coupling_evol.agent.components.internal_model import forward_model as FM
import numpy as np
from coupling_evol.engine.embedded_control import EmbeddedTargetParameter
from typing import List
from scipy.stats import norm
import coupling_evol.engine.common as C
import logging

LOG = logging.getLogger(__name__)

import coupling_evol.data_process.inprocess.recorder as rlog
from coupling_evol.engine import common as C
REC = rlog.get_recorder(prefix=C.RecordNamespace.LIFE_CYCLE.key)


class ValidEmbeddingAggregator(object):
    def __init__(self, motor_dimension: int, sensory_dimension: int, phase_dimension: int, buffer_gait_size: int):
        """
        Aggregates data for FM.MultiPhaseModel::predict, and comparison between ground and predictions.
        The input embeddings each call are saved per segment.

        Phase wise it should look like this (numbers represent sensory phase while letter the motor phase).

           412341234
        ABCDabcdABCD

        Note first three sensor-phase-segments will be not stored as they can't be predicted without whole gait.
        The buffer then saves following:

        sensory: [4,     1,      2,      3]
        motor:  [[ABCD],[aBCD], [abCD], [abcD]]
        phase:  [ph4,    ph1    ,   ph2 , ph3]

        @param motor_dimension:
        @type motor_dimension:
        @param sensory_dimension:
        @type sensory_dimension:
        @param phase_dimension:
        @type phase_dimension:
        @param buffer_gait_size:
        @type buffer_gait_size:
        """
        self.buffer_segment_size = phase_dimension * buffer_gait_size
        self.phase_dimension = phase_dimension
        self.motor_dimension = motor_dimension
        self.sensory_dimension = sensory_dimension
        self.buffer_sensory_phase_segments = np.zeros((self.buffer_segment_size, sensory_dimension))
        self.buffer_gaits = np.zeros((self.buffer_segment_size, 1, motor_dimension, phase_dimension))
        self.buffer_phases = np.zeros((self.buffer_segment_size, phase_dimension))
        ##
        self._current_valid_gait = np.zeros((motor_dimension, phase_dimension))

        self.segment_head = self.buffer_segment_size - 1
        self.last_motion_phase = -1
        self.starting_motion_phase = -1
        self.segments_filled = 0

    def chronologically_ordered_data(self):
        """

        @return: gaits, sensory_phase_segments, phases
        @rtype:
        """
        if self.segments_filled < self.buffer_segment_size:
            return self.buffer_gaits[:self.segments_filled], \
                self.buffer_sensory_phase_segments[:self.segments_filled], \
                self.buffer_phases[:self.segments_filled]
        else:
            segment_tail = (self.segment_head + 1) % self.buffer_segment_size
            segment_tail_size = self.buffer_segment_size - segment_tail

            sensory_phase_segments = np.zeros((self.buffer_segment_size, self.sensory_dimension)) - 666
            gaits = np.zeros((self.buffer_segment_size, 1, self.motor_dimension, self.phase_dimension))
            phases = np.zeros((self.buffer_segment_size, self.phase_dimension))
            ##
            sensory_phase_segments[:segment_tail_size, :] = self.buffer_sensory_phase_segments[segment_tail:, :]
            gaits[:segment_tail_size, :, :, :] = self.buffer_gaits[segment_tail:, :, :, :]
            phases[:segment_tail_size, :] = self.buffer_phases[segment_tail:, :]

            if segment_tail_size < self.buffer_segment_size:
                sensory_phase_segments[segment_tail_size:, :] = self.buffer_sensory_phase_segments[:segment_tail, :]
                gaits[segment_tail_size:, :, :, :] = self.buffer_gaits[:segment_tail, :, :, :]
                phases[segment_tail_size:, :] = self.buffer_phases[:segment_tail, :]

            return gaits, sensory_phase_segments, phases

    def __call__(self, sensory_embedding: np.ndarray, motor_embedding: np.ndarray, motion_phase: np.ndarray):
        phn = np.argmax(motion_phase)
        if self.last_motion_phase == -1:
            self.last_motion_phase = (phn - 1) % self.phase_dimension
        if self.last_motion_phase != phn:
            self.segment_head = (self.segment_head + 1) % self.buffer_segment_size
            if self.segments_filled < self.buffer_segment_size:
                self.segments_filled += 1
            self.buffer_sensory_phase_segments[self.segment_head, :] = sensory_embedding[:, self.last_motion_phase]
            self._current_valid_gait[:, self.last_motion_phase] = motor_embedding[:, self.last_motion_phase]
            self.buffer_gaits[self.segment_head, 0, :, :] = self._current_valid_gait
            self.buffer_phases[self.segment_head, :] = 0
            self.buffer_phases[self.segment_head, self.last_motion_phase] = 1
            self.last_motion_phase = phn
            return True
        else:
            return False


class EstimatorTwoStageCompetition(EnsembleDynamics):
    STD_MIN_MAX = (0.0001, 10)
    ##
    R_FIRST_STAGE_SCORE = "score_1"
    R_SECOND_STAGE_SCORE = "score_2"
    R_ZERO_NEGENTROPY = "ne_zero"
    R_BEST_NEGENTROPY = "ne_best"
    R_MODEL_LOGODDS = "logodd_m"
    R_MODEL_STDS = "std_m"
    R_ZERO_STDS = "std_zero"
    R_IS_DATADRIVEN_DECISION = "datadriven"
    R_MODEL_SELECTION = "m_sel"
    R_BEST_ZERO_LOGODDS = "logodd_bz"
    R_CONFIDENCE_AGGREGATE = "conf_bz"

    ##
    def __init__(self, sensory_dim, motor_dim, phase_dim, models: List[FM.MultiPhaseModel],
                 zero_neighbourhood_epsilon: float, log_best_zero_odds_threshold=0.01, min_confident_elements_rate=0.7,
                 zero_model_standard_deviation=3, gait_buffer_size=10,
                 zero_model_flip_threshold_in=(3, 5), max_performing_periods=np.inf
                 ):
        """

        @param models:
        @type models:
        @param zero_neighbourhood_epsilon:
        @param log_best_zero_odds_threshold: How many "bits" must be the best_model better than zero_model.
        Example: P(d_y=positive|M_best) = 1/2 and P(d_y=positive|M_zero) = 1/4 , then log_2(1/4)-log_2(1/16) = 2
        if the threshold is 1 then M_best should be more than two times better (which it is)
        if the threshold is 2, then M_best should be more than four times better which it is not.
        Although, implementation-wise it is not log2 but log so log(1/2)=-0.7, the value should be in [0, 0.7),
        where 0.7 can be achieved by super-precise model (log(1)=0, which is basically impossible).
        @param max_performing_periods how many performing periods until learning is forced
        """
        super().__init__(sensory_dim, motor_dim, phase_dim, models)

        self._models = models
        self._zero_neighbourhood_epsilon = zero_neighbourhood_epsilon
        self._zero_model_init_standard_deviation = zero_model_standard_deviation
        self._zero_model_deviation = np.zeros((self.sensory_dim, self.phase_dim)) + zero_model_standard_deviation

        self._log_best_zero_odds_threshold = log_best_zero_odds_threshold
        self._min_confident_elements_rate = min_confident_elements_rate
        self.phase_dim = phase_dim
        self.aggr = ValidEmbeddingAggregator(
            motor_dimension=motor_dim, sensory_dimension=sensory_dim, phase_dimension=phase_dim,
            buffer_gait_size=gait_buffer_size)
        self._model_prediction_deviations = [self._std_nrm(np.sqrt(m.d_yu_variance)) for m in self._models]
        self._segment_counter = 0
        self._gait_counter = 0
        self._gait_buffer_size = gait_buffer_size
        ##
        self._log_tmp = {}
        ##
        self._current_model_selection = self.ZERO_MODEL_ID
        self._zero_model_suggestion_history = np.zeros((zero_model_flip_threshold_in[1],))
        self._zero_model_suggestion_history_head = 0
        self._zero_model_suggestion_threshold = zero_model_flip_threshold_in[0]
        ##
        self._motion_model_suggestion_window = np.zeros((20, len(models)))
        if len(models) > 0:
            self._motion_model_suggestion_window[:, -1] = 1
        self._motion_model_suggestion_window_head = 0
        ##
        self._last_period_learning_decision = 0
        self.max_performing_periods = max_performing_periods

    @classmethod
    def _std_nrm(cls, stds: np.ndarray):
        return np.clip(stds, a_min=cls.STD_MIN_MAX[0], a_max=cls.STD_MIN_MAX[1])

    @staticmethod
    def d_y_predicted(gaits: np.ndarray, phases: np.ndarray, models: List[FM.MultiPhaseModel]) -> List[np.ndarray]:
        ret = []
        for m in models:
            preds = m.predict(gaits, phases)
            ret.append(preds[m.phase_n:] - preds[:-m.phase_n])
        return ret

    @staticmethod
    def model_log_odds(d_y_predicted, d_y_predicted_std, d_grounds):
        """

        @param d_y_predicted: (model_n, data_n, sens_n, phase_n)
        @param d_y_predicted_std: (model_n, sens_n, phase_n)
        @param d_grounds:  (sens_n, phase_n)
        @return: (model_n, data_n, sens_n, phase_n)
        """
        return [
            - 0.5 * np.square((d_grounds - d_y_predicted[i]) / d_y_predicted_std[i])
            - np.log(d_y_predicted_std[i]) for i in range(len(d_y_predicted))
        ]

    @staticmethod
    def d_y_eps_cdf(d_y: np.ndarray, d_y_std: np.ndarray, zero_epsilon: np.ndarray):
        """

        @param d_y: (data_n, sens_n, phase_n)
        @param d_y_std: (sens_n, phase_n)
        @param zero_epsilon: (sens_n, phase_n)
        @return: (data_n, sens_n, phase_n), (data_n, sens_n, phase_n)
        @rtype:
        """
        neg_eps = np.zeros(d_y.shape)
        pos_eps = np.zeros(d_y.shape)
        for i in range(len(d_y)):
            neg_eps[i, :, :] = norm.cdf(-zero_epsilon, d_y[i], d_y_std)
            pos_eps[i, :, :] = norm.cdf(+zero_epsilon, d_y[i], d_y_std)

        return neg_eps, pos_eps

    @classmethod
    def d_y_hist(cls, d_y: np.ndarray, d_y_std: np.ndarray, zero_epsilon: np.ndarray):
        """
        Transforms predicted sensory changes d_y into triplets of probabilities [under_eps, in_eps, over_eps].

        @param d_y: (data_n, sens_n, phase_n)
        @param d_y_std: (sens_n, phase_n)
        @param zero_epsilon: (sens_n, phase_n)
        @return: (data_n, sens_n, phase_n), (data_n, sens_n, phase_n)
        @rtype: (data_n, sens_n, phase_n, 3)
        """
        neg_eps, pos_eps = cls.d_y_eps_cdf(d_y, d_y_std, zero_epsilon)
        hst = np.zeros((d_y.shape[0], d_y.shape[1], d_y.shape[2], 3))
        hst[:, :, :, 0] = neg_eps
        hst[:, :, :, 1] = pos_eps - neg_eps
        hst[:, :, :, 2] = 1 - pos_eps
        return hst

    @staticmethod
    def d_y_dirac(d_y: np.ndarray, zero_epsilon: np.ndarray):
        """

        @param d_y: (data_n, sens_n, phase_n)
        @param d_y_std: (sens_n, phase_n)
        @param zero_epsilon: (sens_n, phase_n)
        @return: (data_n, sens_n, phase_n, 3)
        """
        hst = np.zeros((d_y.shape[0], d_y.shape[1], d_y.shape[2], 3))
        hst[:, :, :, 0] = d_y < -zero_epsilon[None, :, :]
        hst[:, :, :, 2] = d_y > zero_epsilon[None, :, :]
        hst[:, :, :, 1] = 1 - hst[:, :, :, 0] - hst[:, :, :, 2]
        return hst

    @classmethod
    def d_y_dirac_hist_entropy(cls, d_y_dirac, d_y_hist):
        """

        @param d_y_dirac: (data_n, sens_n, phase_n, 3)
        @param d_y_hist: (data_n, sens_n, phase_n, 3)
        @return: (data_n, sens_n, phase_n)
        """
        return np.log(np.maximum(np.sum(d_y_dirac * d_y_hist, axis=3), cls.STD_MIN_MAX[0]))

    @staticmethod
    def d_y_mem_to_emb(d_y_mem, d_phase):
        ph_dim = d_phase.shape[1]
        embs = np.zeros((len(d_y_mem) // ph_dim, d_y_mem.shape[1], ph_dim))
        for i in range((len(d_y_mem) // ph_dim)):
            embs[i, :, :] = FM.get_embedding(d_y_mem, d_phase, upper_bound_index=(i + 1) * ph_dim)
        return embs

    def update_model_prediction_deviations(self, d_grounds, dy_predictions, update_alpha=0.05):
        for i in range(len(dy_predictions)):
            model_deviaitons = np.std(d_grounds - dy_predictions[i], axis=0)
            self._model_prediction_deviations[i] = self._std_nrm(
                model_deviaitons * update_alpha + self._model_prediction_deviations[i] * (1 - update_alpha)
            )
        REC(self.R_MODEL_STDS, np.asarray(self._model_prediction_deviations))
        ## zero model!
        zero_model_deviation = np.std(d_grounds, axis=0)
        self._zero_model_deviation = self._std_nrm(
            zero_model_deviation * update_alpha + self._zero_model_deviation * (1 - update_alpha))
        self._zero_model_deviation = np.maximum(self._zero_model_deviation, self.STD_MIN_MAX[0] * 1.1)
        REC(self.R_ZERO_STDS, self._zero_model_deviation)

    def two_stage_comparison(self,
                             dy_grounds: np.ndarray,
                             dy_predictions: List[np.ndarray],
                             model_prediction_deviations: List[np.ndarray]):
        ## COMPARISON BETWEEN MOTION-MODELS
        # if len(dy_predictions) > 1:
        log_odds = self.model_log_odds(dy_predictions, d_y_predicted_std=model_prediction_deviations,
                                       d_grounds=dy_grounds)
        REC(self.R_MODEL_LOGODDS, np.asarray([lo[-1, :, :] for lo in log_odds]))
        neg_loss = [np.sum(lo[-1, :, :]) for lo in log_odds]
        REC(self.R_FIRST_STAGE_SCORE, np.asarray(neg_loss))
        best_model_id = np.argmax(neg_loss)
        _zero_epsilon = self._std_nrm(
            np.sqrt(self._models[best_model_id].d_yu_variance)) * self._zero_neighbourhood_epsilon
        ## COMPARISON WITH ZERO-MODEL
        dy_pb_is_zero = np.zeros(dy_predictions[best_model_id].shape)
        dy_pb_is_zero[(_zero_epsilon - np.abs(dy_predictions[best_model_id])) > 0] = 1.
        ## Outside (motion) comparison
        dy_pb = dy_predictions[best_model_id] * (1 - dy_pb_is_zero)
        d_y_pred_hist = self.d_y_hist(
            d_y=dy_pb, d_y_std=model_prediction_deviations[best_model_id],
            zero_epsilon=_zero_epsilon)
        _d_y_zero_hist = self.d_y_hist(
            d_y=np.zeros((len(d_y_pred_hist), self.sensory_dim, self.phase_dim)),
            d_y_std=self._zero_model_deviation,
            zero_epsilon=_zero_epsilon)
        d_ground_dirac = self.d_y_dirac(dy_grounds, zero_epsilon=_zero_epsilon)
        # getting (neg) entropies
        best_model_entropy_out = self.d_y_dirac_hist_entropy(d_ground_dirac, d_y_pred_hist)
        REC(self.R_BEST_NEGENTROPY, best_model_entropy_out[-1])
        zero_model_entropy_out = self.d_y_dirac_hist_entropy(
            d_ground_dirac, _d_y_zero_hist)
        REC(self.R_ZERO_NEGENTROPY, zero_model_entropy_out[-1])
        ## Inside (non-motion) comparison
        best_model_entropy_in = -np.log(model_prediction_deviations[best_model_id])
        zero_model_entropy_in = -np.log(self._zero_model_deviation)

        ## Combine
        in_cmp = dy_pb_is_zero * d_ground_dirac[:, :, :, 1]
        # calculating log odds: log(P(y|best)/P(y|zero))
        log_best_zero_odds_out = best_model_entropy_out[-1] - zero_model_entropy_out[-1]
        log_best_zero_odds_in = best_model_entropy_in - zero_model_entropy_in
        log_best_zero_odds = log_best_zero_odds_out * (1 - in_cmp[-1]) + log_best_zero_odds_in * in_cmp[-1]
        REC(self.R_BEST_ZERO_LOGODDS, log_best_zero_odds)

        score = log_best_zero_odds >= 0
        # the model is good if it predicts well and confidently on most sensory phase-modalities
        is_best_model_good = np.average(score) > self._min_confident_elements_rate
        REC(self.R_SECOND_STAGE_SCORE, np.average(score))
        ##
        ## DECISION
        if is_best_model_good:
            return best_model_id
        else:
            return self.ZERO_MODEL_ID

    def model_selection(self,
                        gaits: np.ndarray,
                        grounds: np.ndarray,
                        phases: np.ndarray,
                        models: List[FM.MultiPhaseModel]):
        if len(models) == 0:
            return self.ZERO_MODEL_ID

        _d_grounds = grounds[self.phase_dim:] - grounds[:-self.phase_dim]
        _dy_p = self.d_y_predicted(gaits, phases, models)
        d_phase = phases[self.phase_dim:]

        d_grounds = self.d_y_mem_to_emb(_d_grounds, d_phase)
        dy_p = [self.d_y_mem_to_emb(dy, d_phase) for dy in _dy_p]
        ##
        self.update_model_prediction_deviations(d_grounds=d_grounds, dy_predictions=dy_p)
        return self.two_stage_comparison(
            dy_grounds=d_grounds, dy_predictions=dy_p, model_prediction_deviations=self._model_prediction_deviations)

    def _data_driven_suggestion(self) -> int:
        gaits, sensory_vecs, phases = self.aggr.chronologically_ordered_data()
        if self._gait_counter < self._gait_buffer_size:  # there is still the first invalid gait
            gaits = gaits[self.phase_dim:]
            sensory_vecs = sensory_vecs[self.phase_dim:]
            phases = phases[self.phase_dim:]
        return self.model_selection(gaits=gaits, grounds=sensory_vecs, phases=phases, models=self._models)

    def _prior_driven_suggestion(self, motor_embedding: np.ndarray) -> int:
        if len(self._models) == 0:
            return self.ZERO_MODEL_ID
        # return np.argmin([np.sum(np.square(m.u_mean - motor_embedding)) for m in self._models])
        return len(self._models) - 1

    def push_suggestion(self, suggestion: int) -> int:
        self._zero_model_suggestion_history[self._zero_model_suggestion_history_head] = int(
            suggestion == self.ZERO_MODEL_ID)
        self._zero_model_suggestion_history_head = (self._zero_model_suggestion_history_head + 1) % len(
            self._zero_model_suggestion_history)

        if suggestion != self.ZERO_MODEL_ID:
            self._motion_model_suggestion_window[self._motion_model_suggestion_window_head, suggestion] = 1
            self._motion_model_suggestion_window_head = (self._motion_model_suggestion_window_head + 1) % len(
                self._motion_model_suggestion_window)

        if np.sum(self._zero_model_suggestion_history) >= self._zero_model_suggestion_threshold or len(
                self._models) == 0:
            self._zero_model_suggestion_history *= 0
            return self.ZERO_MODEL_ID
        else:
            return np.argmax(np.sum(self._motion_model_suggestion_window, axis=0))

    def __call__(self, sensory_embedding: np.ndarray, motor_embedding: np.ndarray,
                 target_parameter: EmbeddedTargetParameter, motion_phase: np.ndarray):
        if self._gait_counter == 0 and self._segment_counter == 0:  # if in initial state it should decide
            _should_decide = True
        else:
            _should_decide = False

        is_phase_switch = self.aggr(sensory_embedding=sensory_embedding, motor_embedding=motor_embedding,
                                    motion_phase=motion_phase)
        if is_phase_switch:
            self._segment_counter += 1
            if self._segment_counter % self.phase_dim == 0:
                _should_decide = True
                self._segment_counter = 0
                # if self._gait_counter < self._gait_buffer_size:
                self._gait_counter += 1

        if _should_decide:
            if self._gait_counter >= self._gait_buffer_size:  # We need data from at least two gaits, where the first gait is invalid.
                _suggestion = self._data_driven_suggestion()
                REC(self.R_IS_DATADRIVEN_DECISION, 1.)
                if (self._gait_counter - self._last_period_learning_decision) > self.max_performing_periods:
                    suggestion = self.ZERO_MODEL_ID
                else:
                    suggestion = _suggestion
            else:
                suggestion = self._prior_driven_suggestion(motor_embedding)
                REC(self.R_IS_DATADRIVEN_DECISION, 0.)

            self._current_model_selection = self.push_suggestion(suggestion)
            if self._current_model_selection == self.ZERO_MODEL_ID:
                self._last_period_learning_decision = self._gait_counter

        REC(self.R_MODEL_SELECTION, self._current_model_selection)
        return self._current_model_selection


class TwoStageAggregatedScoreCompetition(EstimatorTwoStageCompetition):

    def __init__(
            self, sensory_dim, motor_dim, phase_dim, models: List[FM.MultiPhaseModel],
            zero_neighbourhood_epsilon: float,
            log_best_zero_odds_threshold=0.01, min_confident_elements_rate=0.7,
            zero_model_standard_deviation=3, gait_buffer_size=10,
            ensemble_confidence_lr=0.3, log_ensemble_confidence_combiner=True,
            max_performing_periods=np.inf, force_last_model_selection=False
    ):
        super().__init__(
            sensory_dim, motor_dim, phase_dim, models, zero_neighbourhood_epsilon,
            log_best_zero_odds_threshold=log_best_zero_odds_threshold,
            min_confident_elements_rate=min_confident_elements_rate,
            zero_model_flip_threshold_in=(2, 3), gait_buffer_size=gait_buffer_size,
            zero_model_standard_deviation=zero_model_standard_deviation,
            max_performing_periods=max_performing_periods
        )
        if log_ensemble_confidence_combiner:
            self.motion_models_confidence = np.zeros((self.sensory_dim, self.phase_dim)) + 0.05
        else:
            self.motion_models_confidence = np.zeros((self.sensory_dim, self.phase_dim)) + 0.6

        self._models_confidence_lr = ensemble_confidence_lr
        self._log_ensemble_confidence_combiner = log_ensemble_confidence_combiner
        self._models_logodds = [-np.log(self._std_nrm(np.sqrt(m.d_yu_variance))) for m in models]
        self.current_target_parameter = EmbeddedTargetParameter.simple_target(np.zeros((sensory_dim, phase_dim)))
        self.force_last_model = force_last_model_selection

    @property
    def _motion_models_confidence(self):
        return self.motion_models_confidence

    @_motion_models_confidence.setter
    def _motion_models_confidence(self, value):
        self.motion_models_confidence = value

    def _update_model_log_odds(self, log_odds):
        for i in range(len(self._models)):
            self._models_logodds[i] = log_odds[i] * self._models_confidence_lr + \
                                      self._models_logodds[i] * (1 - self._models_confidence_lr)
        REC(self.R_MODEL_LOGODDS, np.asarray(self._models_logodds))

    def _first_stage_comparison(self,
                                dy_grounds: np.ndarray,
                                dy_predictions: List[np.ndarray],
                                model_prediction_deviations: List[np.ndarray],
                                target_weight: np.ndarray,
                                ):
        log_odds = self.model_log_odds(dy_predictions, d_y_predicted_std=model_prediction_deviations,
                                       d_grounds=dy_grounds)
        self._update_model_log_odds([lo[-1] for lo in log_odds])
        neg_loss = [np.sum(lo * target_weight) for lo in self._models_logodds]
        REC(self.R_FIRST_STAGE_SCORE, np.asarray(neg_loss))
        return np.argmax(neg_loss)

    def _second_stage_comparison(self,
                                 dy_grounds: np.ndarray,
                                 dy_prediction: np.ndarray,
                                 model_prediction_deviation: np.ndarray,
                                 target_weight: np.ndarray,
                                 zero_epsilon: np.ndarray,
                                 ):
        ## COMPARISON WITH ZERO-MODEL
        # dy_pb_is_zero = np.zeros(dy_prediction.shape)
        # dy_pb_is_zero[(zero_epsilon - np.abs(dy_prediction)) > 0] = 1.
        ## Outside (motion) comparison
        dy_pb = dy_prediction  # * (1 - dy_pb_is_zero)
        d_y_pred_hist = self.d_y_hist(
            d_y=dy_pb, d_y_std=model_prediction_deviation,
            zero_epsilon=zero_epsilon)
        _d_y_zero_hist = self.d_y_hist(
            d_y=np.zeros((len(d_y_pred_hist), self.sensory_dim, self.phase_dim)),
            d_y_std=self._zero_model_deviation,
            zero_epsilon=zero_epsilon)
        d_ground_dirac = self.d_y_dirac(dy_grounds, zero_epsilon=zero_epsilon)
        # getting (neg) entropies
        best_model_entropy_out = self.d_y_dirac_hist_entropy(d_ground_dirac, d_y_pred_hist)
        REC(self.R_BEST_NEGENTROPY, best_model_entropy_out[-1])
        zero_model_entropy_out = self.d_y_dirac_hist_entropy(
            d_ground_dirac, _d_y_zero_hist)
        REC(self.R_ZERO_NEGENTROPY, zero_model_entropy_out[-1])
        ## Inside (non-motion) comparison
        # best_model_entropy_in = -np.log(model_prediction_deviation)
        # zero_model_entropy_in = -np.log(self._zero_model_deviation)

        ## Combine
        # in_cmp = dy_pb_is_zero * d_ground_dirac[:, :, :, 1]
        # calculating log odds: log(P(y|best)/P(y|zero))
        log_best_zero_odds_out = best_model_entropy_out[-1] - zero_model_entropy_out[-1]
        # log_best_zero_odds_in = best_model_entropy_in - zero_model_entropy_in

        log_best_zero_odds = log_best_zero_odds_out  # * (1 - in_cmp[-1]) + log_best_zero_odds_in * in_cmp[-1]
        REC(self.R_BEST_ZERO_LOGODDS, log_best_zero_odds)

        if self._log_ensemble_confidence_combiner:
            score = log_best_zero_odds
            self._motion_models_confidence = score * (self._models_confidence_lr) + \
                                             self._motion_models_confidence * (1 - self._models_confidence_lr)
            is_confident_element = self._motion_models_confidence >= 0
        else:
            score = log_best_zero_odds >= 0
            self._motion_models_confidence = score * (self._models_confidence_lr) + \
                                             self._motion_models_confidence * (1 - self._models_confidence_lr)
            is_confident_element = self._motion_models_confidence >= 0.5
        REC(self.R_CONFIDENCE_AGGREGATE, self._motion_models_confidence)
        # the model is good if it predicts well and confidently on most sensory phase-modalities

        weighted_score = np.sum(is_confident_element * target_weight) / np.sum(target_weight)
        REC(self.R_SECOND_STAGE_SCORE, weighted_score)
        is_best_model_good = weighted_score > self._min_confident_elements_rate
        return is_best_model_good

    def two_stage_comparison(self,
                             dy_grounds: np.ndarray,
                             dy_predictions: List[np.ndarray],
                             model_prediction_deviations: List[np.ndarray]):
        ## COMPARISON BETWEEN MOTION-MODELS
        # if len(dy_predictions) > 1:
        target_weight = self.current_target_parameter.weight
        best_model_id = self._first_stage_comparison(
            dy_grounds, dy_predictions, model_prediction_deviations, target_weight)
        if self.force_last_model:
            best_model_id = len(self._models) - 1

        _zero_epsilon = self._std_nrm(
            np.sqrt(self._models[best_model_id].d_yu_variance)) * self._zero_neighbourhood_epsilon

        is_best_model_good = self._second_stage_comparison(
            dy_grounds=dy_grounds,
            dy_prediction=dy_predictions[best_model_id],
            model_prediction_deviation=model_prediction_deviations[best_model_id],
            target_weight=target_weight,
            zero_epsilon=_zero_epsilon
        )
        ## DECISION
        if is_best_model_good:
            return best_model_id
        else:
            return self.ZERO_MODEL_ID

    def push_suggestion(self, suggestion: int) -> int:
        return suggestion

    def __call__(self, sensory_embedding: np.ndarray, motor_embedding: np.ndarray,
                 target_parameter: EmbeddedTargetParameter, motion_phase: np.ndarray):
        self.current_target_parameter = target_parameter
        return super().__call__(sensory_embedding, motor_embedding, target_parameter, motion_phase)


class TwoStageAggregatedScoreComposition(EstimatorTwoStageCompetition):
    def __init__(
            self, sensory_dim, motor_dim, phase_dim, models: List[FM.MultiPhaseModel],
            zero_neighbourhood_epsilon: float,
            log_best_zero_odds_threshold=0.01, min_confident_elements_rate=0.7,
            zero_model_standard_deviation=3, gait_buffer_size=10,
            ensemble_confidence_lr=0.3, log_ensemble_confidence_combiner=True,
            max_performing_periods=np.inf
    ):
        super().__init__(
            sensory_dim, motor_dim, phase_dim, models, zero_neighbourhood_epsilon,
            log_best_zero_odds_threshold=log_best_zero_odds_threshold,
            min_confident_elements_rate=min_confident_elements_rate,
            zero_model_flip_threshold_in=(2, 3), gait_buffer_size=gait_buffer_size,
            zero_model_standard_deviation=zero_model_standard_deviation,
            max_performing_periods=max_performing_periods
        )
        self._motion_models_confidence = np.zeros((self.sensory_dim, self.phase_dim)) + 1
        self._models_confidence_lr = ensemble_confidence_lr
        self._log_ensemble_confidence_combiner = log_ensemble_confidence_combiner
        self._models_logodds = [-np.log(self._std_nrm(np.sqrt(m.d_yu_variance))) for m in models]
        self.current_target_parameter = EmbeddedTargetParameter.simple_target(np.zeros((sensory_dim, phase_dim)))

        if len(self._models) > 0:
            self.current_compound_model = CompoundModel(self._models, np.asarray(self._models_logodds))
        else:
            self.current_compound_model = None

    def _update_model_log_odds(self, log_odds):
        for i in range(len(self._models)):
            self._models_logodds[i] = log_odds[i] * self._models_confidence_lr + \
                                      self._models_logodds[i] * (1 - self._models_confidence_lr)
        REC(self.R_MODEL_LOGODDS, np.asarray(self._models_logodds))

    def two_stage_comparison(self,
                             dy_grounds: np.ndarray,
                             dy_predictions: List[np.ndarray],
                             model_prediction_deviations: List[np.ndarray]):
        ## COMPARISON BETWEEN MOTION-MODELS
        target_weight = self.current_target_parameter.weight
        log_odds = self.model_log_odds(dy_predictions, d_y_predicted_std=model_prediction_deviations,
                                       d_grounds=dy_grounds)
        self._update_model_log_odds([lo[-1] for lo in log_odds])
        # del self.current_compound_model
        self.current_compound_model = CompoundModel(self._models, np.asarray(self._models_logodds))

        neg_loss = [np.sum(lo * target_weight) for lo in self._models_logodds]
        REC(self.R_FIRST_STAGE_SCORE, np.asarray(neg_loss))
        _zero_epsilon = self._std_nrm(
            np.sqrt(self.current_compound_model.d_yu_variance)) * self._zero_neighbourhood_epsilon
        ## COMPARISON WITH ZERO-MODEL
        _dy_predictions = self.current_compound_model.combine_sensory_embeddings(np.asarray(dy_predictions))

        dy_pb_is_zero = np.zeros(_dy_predictions.shape)
        dy_pb_is_zero[(_zero_epsilon - np.abs(_dy_predictions)) > 0] = 1.
        ## Outside (motion) comparison
        _compound_model_prediction_deviations = np.sqrt(self.current_compound_model.combine_sensory_embeddings(
            np.square(np.asarray(model_prediction_deviations))))
        dy_pb = _dy_predictions * (1 - dy_pb_is_zero)
        d_y_pred_hist = self.d_y_hist(
            d_y=dy_pb, d_y_std=_compound_model_prediction_deviations,
            zero_epsilon=_zero_epsilon)
        _d_y_zero_hist = self.d_y_hist(
            d_y=np.zeros((len(d_y_pred_hist), self.sensory_dim, self.phase_dim)),
            d_y_std=self._zero_model_deviation,
            zero_epsilon=_zero_epsilon)
        d_ground_dirac = self.d_y_dirac(dy_grounds, zero_epsilon=_zero_epsilon)
        # getting (neg) entropies
        best_model_entropy_out = self.d_y_dirac_hist_entropy(d_ground_dirac, d_y_pred_hist)
        REC(self.R_BEST_NEGENTROPY, best_model_entropy_out[-1])
        zero_model_entropy_out = self.d_y_dirac_hist_entropy(
            d_ground_dirac, _d_y_zero_hist)
        REC(self.R_ZERO_NEGENTROPY, zero_model_entropy_out[-1])
        ## Inside (non-motion) comparison
        best_model_entropy_in = -np.log(_compound_model_prediction_deviations)
        zero_model_entropy_in = -np.log(self._zero_model_deviation)

        ## Combine
        # in_cmp = dy_pb_is_zero * d_ground_dirac[:, :, :, 1]
        # calculating log odds: log(P(y|best)/P(y|zero))
        log_best_zero_odds_out = best_model_entropy_out[-1] - zero_model_entropy_out[-1]
        # log_best_zero_odds_in = best_model_entropy_in - zero_model_entropy_in

        log_best_zero_odds = log_best_zero_odds_out  # * (1 - in_cmp[-1]) + log_best_zero_odds_in * in_cmp[-1]
        REC(self.R_BEST_ZERO_LOGODDS, log_best_zero_odds)

        if self._log_ensemble_confidence_combiner:
            score = log_best_zero_odds
            self._motion_models_confidence = score * (self._models_confidence_lr) + \
                                             self._motion_models_confidence * (1 - self._models_confidence_lr)
            is_confident_element = self._motion_models_confidence >= 0
        else:
            score = log_best_zero_odds >= 0
            self._motion_models_confidence = score * (self._models_confidence_lr) + \
                                             self._motion_models_confidence * (1 - self._models_confidence_lr)
            is_confident_element = self._motion_models_confidence >= 0.5

        REC(self.R_CONFIDENCE_AGGREGATE, self._motion_models_confidence)
        # the model is good if it predicts well and confidently on most sensory phase-modalities
        weighted_score = np.sum(is_confident_element * target_weight) / np.sum(target_weight)
        is_best_model_good = weighted_score > self._min_confident_elements_rate
        REC(self.R_SECOND_STAGE_SCORE, weighted_score)
        ## DECISION
        if is_best_model_good:
            return self.current_compound_model.best_model_id
        else:
            return self.ZERO_MODEL_ID

    def push_suggestion(self, suggestion: int) -> int:
        return suggestion

    def __call__(self, sensory_embedding: np.ndarray, motor_embedding: np.ndarray,
                 target_parameter: EmbeddedTargetParameter, motion_phase: np.ndarray):
        self.current_target_parameter = target_parameter
        return super().__call__(sensory_embedding, motor_embedding, target_parameter, motion_phase)


class TwoStageAggregatedScoreCompetitionOdometrySum(TwoStageAggregatedScoreCompetition):
    def __init__(
            self, sensory_dim, motor_dim, phase_dim, models: List[FM.MultiPhaseModel],
            zero_neighbourhood_epsilon: float,
            log_best_zero_odds_threshold=0.01, min_confident_elements_rate=0.7,
            zero_model_standard_deviation=3, gait_buffer_size=100,
            ensemble_confidence_lr=0.3, log_ensemble_confidence_combiner=True,
            max_performing_periods=np.inf, force_last_model_selection=False,
            max_zero_suggestions=30, direct_confidence_score=False, max_candidate_suggestions=10
    ):
        super().__init__(
            sensory_dim, motor_dim, phase_dim, models, zero_neighbourhood_epsilon,
            log_best_zero_odds_threshold=log_best_zero_odds_threshold,
            min_confident_elements_rate=min_confident_elements_rate,
            gait_buffer_size=gait_buffer_size,
            zero_model_standard_deviation=zero_model_standard_deviation,
            max_performing_periods=max_performing_periods,
            ensemble_confidence_lr=ensemble_confidence_lr,
            log_ensemble_confidence_combiner=log_ensemble_confidence_combiner,
            force_last_model_selection=force_last_model_selection

        )

        self._zero_model_deviation = self._std_nrm(
            np.sqrt(self._odometry_sum_transformation(np.square(self._zero_model_deviation))))
        self._model_prediction_deviations = [self._std_nrm(np.sqrt(self._odometry_sum_transformation(m.d_yu_variance)))
                                             for m in self._models]

        # self._model_prediction_deviations = [self._zero_model_deviation for m in self._models]
        # self._models_logodds = [-np.log(self._std_nrm(np.sqrt(self._odometry_sum_transformation(m.d_yu_variance)))) for
        # m in models]
        self._models_logodds = [
            -np.log(self._std_nrm(np.sqrt(self._odometry_sum_transformation(np.ones_like(m.d_yu_variance))))) for
            m in models]
        self.zero_suggestion_counter = 0
        self.max_zero_suggestions = max_zero_suggestions
        self.direct_confidence_score = direct_confidence_score
        self.candidate_model = 0
        self.candidate_suggestion_counter = 0
        self.max_candidate_suggestions = max_candidate_suggestions

    @staticmethod
    def _odometry_sum_transformation(embeddings: np.ndarray):
        """
        @param embeddings: (data_n, sens_n, phase_n)
        @return: (data_n, sens_n, phase_n)
        """
        ret = np.zeros(embeddings.shape)
        if embeddings.ndim == 3:
            ret[:, :5, :] = np.sum(embeddings[:, :5, :], axis=2)[:, :, None]
            ret[:, 5:, :] = embeddings[:, 5:, :]
        else:
            ret[:5, :] = np.sum(embeddings[:5, :], axis=1)[:, None]
            ret[5:, :] = embeddings[5:, :]
        return ret

    def model_selection(self,
                        gaits: np.ndarray,
                        grounds: np.ndarray,
                        phases: np.ndarray,
                        models: List[FM.MultiPhaseModel]):
        if len(models) == 0:
            return self.ZERO_MODEL_ID

        _d_grounds = grounds[self.phase_dim:] - grounds[:-self.phase_dim]
        _dy_p = self.d_y_predicted(gaits, phases, models)
        d_phase = phases[self.phase_dim:]

        d_grounds = self.d_y_mem_to_emb(_d_grounds, d_phase)
        d_grounds = self._odometry_sum_transformation(d_grounds)

        dy_ps = [self.d_y_mem_to_emb(dy, d_phase) for dy in _dy_p]
        dy_ps = [self._odometry_sum_transformation(dy_p) for dy_p in dy_ps]
        ##
        self.update_model_prediction_deviations(d_grounds=d_grounds, dy_predictions=dy_ps,
                                                update_alpha=min(self._gait_counter / self._gait_buffer_size, 1))
        return self.two_stage_comparison(
            dy_grounds=d_grounds, dy_predictions=dy_ps, model_prediction_deviations=self._model_prediction_deviations)

    def push_suggestion(self, suggestion: int) -> int:

        if self._current_model_selection == self.ZERO_MODEL_ID:
            return suggestion

        if suggestion == self.ZERO_MODEL_ID:
            self.zero_suggestion_counter += 1
        else:
            self.zero_suggestion_counter = 0

        if suggestion != self.ZERO_MODEL_ID and suggestion != self.candidate_model:
            self.candidate_model = suggestion
            self.candidate_suggestion_counter = 0
        ##
        self.candidate_suggestion_counter += 1
        ##
        if suggestion != self.ZERO_MODEL_ID and self.candidate_suggestion_counter > self.max_candidate_suggestions:
            return self.candidate_model

        if self.zero_suggestion_counter == self.max_zero_suggestions:
            return self.ZERO_MODEL_ID

        return self._current_model_selection

    def update_model_log_odds(self, log_odds, update_alpha):
        for i in range(len(self._models)):
            self._models_logodds[i] = log_odds[i] * update_alpha + \
                                      self._models_logodds[i] * (1 - update_alpha)
        REC(self.R_MODEL_LOGODDS, np.asarray(self._models_logodds))

    def _second_stage_comparison(self,
                                 dy_grounds: np.ndarray,
                                 dy_prediction: np.ndarray,
                                 model_prediction_deviation: np.ndarray,
                                 target_weight: np.ndarray,
                                 zero_epsilon: np.ndarray,
                                 ):
        zero_epsilon_model = model_prediction_deviation * zero_epsilon
        zero_epsilon_zero = self._zero_model_deviation * zero_epsilon
        ## COMPARISON WITH ZERO-MODEL
        ## Outside (motion) comparison
        dy_pb = dy_prediction
        d_y_pred_hist = self.d_y_hist(
            d_y=dy_pb, d_y_std=model_prediction_deviation,
            zero_epsilon=zero_epsilon_model)
        _d_y_zero_hist = self.d_y_hist(
            d_y=np.zeros((len(d_y_pred_hist), self.sensory_dim, self.phase_dim)),
            d_y_std=self._zero_model_deviation,
            zero_epsilon=zero_epsilon_zero)
        d_ground_dirac_model = self.d_y_dirac(dy_grounds, zero_epsilon=zero_epsilon_model)
        d_ground_dirac_zero = self.d_y_dirac(dy_grounds, zero_epsilon=zero_epsilon_zero)
        # getting (neg) entropies
        best_model_entropy_out = self.d_y_dirac_hist_entropy(d_ground_dirac_model, d_y_pred_hist)
        REC(self.R_BEST_NEGENTROPY, best_model_entropy_out[-1])
        zero_model_entropy_out = self.d_y_dirac_hist_entropy(
            d_ground_dirac_zero, _d_y_zero_hist)
        REC(self.R_ZERO_NEGENTROPY, zero_model_entropy_out[-1])
        ## Inside (non-motion) comparison

        ## Combine
        # calculating log odds: log(P(y|best)/P(y|zero))
        log_best_zero_odds_out = best_model_entropy_out[-1] - zero_model_entropy_out[-1]

        log_best_zero_odds = log_best_zero_odds_out
        REC(self.R_BEST_ZERO_LOGODDS, log_best_zero_odds)

        if self._log_ensemble_confidence_combiner:
            score = log_best_zero_odds
            self._motion_models_confidence = score * (self._models_confidence_lr) + \
                                             self._motion_models_confidence * (1 - self._models_confidence_lr)
            is_confident_element = self._motion_models_confidence >= 0
        else:
            score = log_best_zero_odds >= 0
            self._motion_models_confidence = score * (self._models_confidence_lr) + \
                                             self._motion_models_confidence * (1 - self._models_confidence_lr)
            is_confident_element = self._motion_models_confidence >= 0.5
        REC(self.R_CONFIDENCE_AGGREGATE, self._motion_models_confidence)
        # the model is good if it predicts well and confidently on most sensory phase-modalities
        if not self.direct_confidence_score:
            weighted_score = np.sum(is_confident_element * target_weight) / np.sum(target_weight)
        else:
            weighted_score = np.sum(self._motion_models_confidence * target_weight) / np.sum(target_weight)
        REC(self.R_SECOND_STAGE_SCORE, weighted_score)
        is_best_model_good = weighted_score > self._min_confident_elements_rate
        return is_best_model_good

    def _second_stage_comparison_all(self,
                                     dy_grounds: np.ndarray,
                                     dy_predictions: List[np.ndarray],
                                     model_prediction_deviations: List[np.ndarray],
                                     target_weight: np.ndarray,
                                     zero_epsilon: np.ndarray,
                                     ):
        last_dy_preds = np.asarray([dy_pred[-1] for dy_pred in dy_predictions])
        last_stds = np.asarray([std for std in model_prediction_deviations])
        nrm_preds = last_dy_preds/last_stds
        nrm_grounds = np.asarray([dy_grounds[-1]/std for std in last_stds])
        nrm_zero_ground = dy_grounds[-1]/self._zero_model_deviation
        nrm_weight = target_weight/np.sum(target_weight)
        ## COMPARISON WITH ZERO-MODEL
        ## Outside (motion) comparison
        d_y_pred_hist = self.d_y_hist(d_y=nrm_preds, d_y_std=np.ones_like(zero_epsilon), zero_epsilon=zero_epsilon)
        _d_y_zero_hist = self.d_y_hist(d_y=np.zeros((len(d_y_pred_hist), self.sensory_dim, self.phase_dim)),
            d_y_std=np.ones_like(self._zero_model_deviation), zero_epsilon=zero_epsilon)

        d_ground_dirac_model = self.d_y_dirac(nrm_grounds, zero_epsilon=zero_epsilon)
        d_ground_dirac_zero = self.d_y_dirac(np.asarray([nrm_zero_ground]*len(dy_predictions)), zero_epsilon=zero_epsilon)
        # getting (neg) entropies
        best_model_entropy_out = self.d_y_dirac_hist_entropy(d_ground_dirac_model, d_y_pred_hist)

        zero_model_entropy_out = self.d_y_dirac_hist_entropy(d_ground_dirac_zero, _d_y_zero_hist)
        REC(self.R_ZERO_NEGENTROPY, zero_model_entropy_out[0])
        ## Inside (non-motion) comparison

        ## Combine
        # calculating log odds: log(P(y|best)/P(y|zero))
        log_model_zero_odds = best_model_entropy_out - zero_model_entropy_out

        # FIXME I have to calc best before like this due to comp with visuals
        best = np.argmax(np.sum(log_model_zero_odds * nrm_weight[None, :, :], axis=(1,2)))
        log_best_zero_odds = log_model_zero_odds[best]

        REC(self.R_BEST_NEGENTROPY, best_model_entropy_out[best])
        REC(self.R_BEST_ZERO_LOGODDS, log_best_zero_odds)

        if self._log_ensemble_confidence_combiner:
            score = log_best_zero_odds
            self._motion_models_confidence = score * (self._models_confidence_lr) + \
                                             self._motion_models_confidence * (1 - self._models_confidence_lr)
            is_confident_element = self._motion_models_confidence >= 0
        else:
            score = log_best_zero_odds >= 0
            self._motion_models_confidence = score * (self._models_confidence_lr) + \
                                             self._motion_models_confidence * (1 - self._models_confidence_lr)
            is_confident_element = self._motion_models_confidence >= 0.5
        REC(self.R_CONFIDENCE_AGGREGATE, self._motion_models_confidence)
        # the model is good if it predicts well and confidently on most sensory phase-modalities
        if not self.direct_confidence_score:
            weighted_score = np.sum(is_confident_element * nrm_weight)
        else:
            weighted_score = np.sum(self._motion_models_confidence * nrm_weight)
        REC(self.R_SECOND_STAGE_SCORE, weighted_score)
        is_best_model_good = weighted_score > self._min_confident_elements_rate
        return is_best_model_good

    def _first_stage_comparison(self,
                                dy_grounds: np.ndarray,
                                dy_predictions: List[np.ndarray],
                                model_prediction_deviations: List[np.ndarray],
                                target_weight: np.ndarray,
                                ):
        _target_weight = target_weight/np.sum(target_weight)
        log_odds = self.model_log_odds(dy_predictions, d_y_predicted_std=model_prediction_deviations,
                                       d_grounds=dy_grounds)
        self.update_model_log_odds([np.mean(lo, axis=0) for lo in log_odds],
                                   # update_alpha=min(self._gait_counter / self._gait_buffer_size, 1))
                                   update_alpha=min(self._gait_counter / self._gait_buffer_size, 10*self._models_confidence_lr))
        neg_loss = [np.sum(lo * _target_weight) for lo in self._models_logodds]
        REC(self.R_FIRST_STAGE_SCORE, np.asarray(neg_loss))
        return np.argmax(neg_loss)

    def two_stage_comparison(self,
                             dy_grounds: np.ndarray,
                             dy_predictions: List[np.ndarray],
                             model_prediction_deviations: List[np.ndarray]):

        target_weight = self.current_target_parameter.weight

        best_model_id = self._first_stage_comparison(
            dy_grounds, dy_predictions, model_prediction_deviations, target_weight)
        if self.force_last_model:
            best_model_id = len(self._models) - 1

        # _zero_epsilon = model_prediction_deviations[best_model_id] * self._zero_neighbourhood_epsilon
        _zero_epsilon = np.ones_like(model_prediction_deviations[best_model_id]) * self._zero_neighbourhood_epsilon
        # _zero_epsilon = self._std_nrm(np.sqrt(self._odometry_sum_transformation(self._models[best_model_id].d_yu_variance))) * self._zero_neighbourhood_epsilon

        is_best_model_good = self._second_stage_comparison_all(
            dy_grounds=dy_grounds,
            dy_predictions=dy_predictions,
            model_prediction_deviations=model_prediction_deviations,
            target_weight=target_weight,
            zero_epsilon=_zero_epsilon
        )
        ## DECISION
        if is_best_model_good:
            return best_model_id
        else:
            return self.ZERO_MODEL_ID


class TwoStageAggregatedScoreCompositionOdometrySum(TwoStageAggregatedScoreCompetitionOdometrySum):
    R_MODEL_MODALITY_PHASE_WEIGHTS = 'model_modality_phase_weights' 
    

    def __init__(
            self, sensory_dim, motor_dim, phase_dim, models: List[FM.MultiPhaseModel],
            zero_neighbourhood_epsilon: float,
            log_best_zero_odds_threshold=0.01, min_confident_elements_rate=0.7,
            zero_model_standard_deviation=3, gait_buffer_size=100,
            ensemble_confidence_lr=0.3, log_ensemble_confidence_combiner=True,
            max_performing_periods=np.inf, force_last_model_selection=False,
            max_zero_suggestions=30, direct_confidence_score=False, max_candidate_suggestions=10,
            composite_weight_normer='softmax', softmax_power=1, continual_model_update=True, submodel_combination=True
    ):
        super().__init__(
            sensory_dim, motor_dim, phase_dim, models, zero_neighbourhood_epsilon,
            log_best_zero_odds_threshold=log_best_zero_odds_threshold,
            min_confident_elements_rate=min_confident_elements_rate,
            gait_buffer_size=gait_buffer_size,
            zero_model_standard_deviation=zero_model_standard_deviation,
            max_performing_periods=max_performing_periods,
            ensemble_confidence_lr=ensemble_confidence_lr,
            log_ensemble_confidence_combiner=log_ensemble_confidence_combiner,
            force_last_model_selection=force_last_model_selection,
            max_zero_suggestions=max_zero_suggestions, direct_confidence_score=direct_confidence_score,
            max_candidate_suggestions=max_candidate_suggestions
        )
        if len(self._models) > 0:
            self.current_compound_model = CompoundModel(self._models, np.asarray(self._models_logodds))
            self._candidate_compound_model = CompoundModel(self._models, np.asarray(self._models_logodds))
        else:
            self.current_compound_model = None
            self._candidate_compound_model = None

        # select normer
        if composite_weight_normer == 'softmax':
            self._normer = lambda x: self._softmax_normer(x, power=softmax_power)
        elif composite_weight_normer == 'one_hot':
            self._normer = self._one_hot_normer
        else:
            raise ValueError(f'Unknown normer: {composite_weight_normer}')
        
        self._slow_model_logodds = np.asarray(self._models_logodds)
        self.continual_model_update = continual_model_update
        self.submodel_combination = submodel_combination
        
    @staticmethod
    def _softmax_normer(model_logodds: np.array, power=1):
        shift = model_logodds - np.max(model_logodds, axis=0)
        exp_logodds = np.exp(shift * power)
        return exp_logodds / np.sum(exp_logodds, axis=0)
    
    @staticmethod
    def _one_hot_normer(model_logodds: np.array):
        return one_hot_matrix(np.argmax(model_logodds, axis=0), model_logodds.shape[0])
    
    def slow_update_model_log_odds(self, models_log_odds: np.ndarray, update_alpha):
        self._slow_model_logodds = models_log_odds * update_alpha + \
                              self._slow_model_logodds * (1 - update_alpha)


    def _first_stage_comparison(self,
                                dy_grounds: np.ndarray,
                                dy_predictions: List[np.ndarray],
                                model_prediction_deviations: List[np.ndarray],
                                target_weight: np.ndarray,
                                ):     
        _target_weight = target_weight/np.sum(target_weight)
        log_odds = self.model_log_odds(dy_predictions, d_y_predicted_std=model_prediction_deviations,
                                       d_grounds=dy_grounds)
        self.update_model_log_odds([np.mean(lo, axis=0) for lo in log_odds],
                                   update_alpha=min(self._gait_counter / self._gait_buffer_size, 10*self._models_confidence_lr))
        neg_loss = [np.sum(lo * _target_weight) for lo in self._models_logodds]

        REC(self.R_FIRST_STAGE_SCORE, np.asarray(neg_loss))

        # self.slow_update_model_log_odds(np.asarray(self._models_logodds), self._models_confidence_lr * 0.1)
        if not self.submodel_combination:
            weighted_logodds = np.ones((len(self._models), self.sensory_dim, self.phase_dim)) * np.sum(np.asarray(self._models_logodds) * _target_weight[None, :, :], axis=(1,2))[:, None, None]
        else:
            weighted_logodds = np.asarray(self._models_logodds) 
        model_modality_phase_weights = self._normer(weighted_logodds)
        REC(self.R_MODEL_MODALITY_PHASE_WEIGHTS, model_modality_phase_weights)

        _candidate_compound_model = CompoundModel(self._models, model_modality_phase_weights=model_modality_phase_weights)
        if self.continual_model_update:
            self.current_compound_model = _candidate_compound_model
            self._candidate_compound_model = _candidate_compound_model
        else:
            self.current_compound_model = self.current_compound_model
            self._candidate_compound_model = _candidate_compound_model

        return np.argmax(neg_loss)

    def _prior_driven_suggestion(self, motor_embedding: np.ndarray) -> int:
        ret = super()._prior_driven_suggestion(motor_embedding)
        if ret != self.ZERO_MODEL_ID:
            prior_weights = np.zeros((len(self._models), self.sensory_dim, self.phase_dim))
            prior_weights[ret, :, :] = 1
            self.current_compound_model = CompoundModel(self._models, model_modality_phase_weights=prior_weights)
        return ret

    def push_suggestion(self, suggestion: int) -> int:
    
            if self._current_model_selection == self.ZERO_MODEL_ID:
                return suggestion
    
            if suggestion == self.ZERO_MODEL_ID:
                self.zero_suggestion_counter += 1
            else:
                self.zero_suggestion_counter = 0
    
            if suggestion != self.ZERO_MODEL_ID and suggestion != self.candidate_model:
                self.candidate_model = suggestion
                self.candidate_suggestion_counter = 0
            ##
            self.candidate_suggestion_counter += 1
            ##
            if suggestion != self.ZERO_MODEL_ID and self.candidate_suggestion_counter > self.max_candidate_suggestions:
                if self.candidate_model != self._current_model_selection:
                    self.current_compound_model = self._candidate_compound_model
                return self.candidate_model
    
            if self.zero_suggestion_counter == self.max_zero_suggestions:
                return self.ZERO_MODEL_ID
    
            return self._current_model_selection

if __name__ == '__main__':
    print(EstimatorTwoStageCompetition.d_y_hist(np.zeros((1, 5, 6)), np.zeros((5, 6)) + 0.1, 0.3)[0])
    ValidEmbeddingAggregator(2, 2, 4, 5)
