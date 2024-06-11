from coupling_evol.agent.components.internal_model import forward_model as FM
import numpy as np
from typing import List
import logging
from coupling_evol.agent.components.ensemble_dynamics.estimator_competition import \
    TwoStageAggregatedScoreCompetitionOdometrySum

LOG = logging.getLogger(__name__)


class SubmodesModelSelection(TwoStageAggregatedScoreCompetitionOdometrySum):
    STD_MIN_MAX = (1, 100)

    def __init__(
            self, sensory_dim, motor_dim, phase_dim, models: List[FM.MultiPhaseModel],
            zero_neighbourhood_epsilon: float,
            log_best_zero_odds_threshold=0.01, min_confident_elements_rate=0.7,
            zero_model_standard_deviation=3, gait_buffer_size=100,
            ensemble_confidence_lr=0.3, log_ensemble_confidence_combiner=True,
            max_performing_periods=np.inf, force_last_model_selection=False,
            min_max_selection_eval_time=(10, 100)
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
        self.error_models_stdv = [np.sum(self._std_nrm(np.sqrt(self._odometry_sum_transformation(m.d_yu_variance)))) for
                                  m in self._models]
        self.error_models_mean = [20. for m in self._models]

        self._models_logodds = [0 for m in models]  # not logodds anymore, just normed errors
        self._model_signed_errors = [1 for m in models]  # not logodds anymore, just normed errors

        self._motion_models_confidence = np.zeros((self.sensory_dim, self.phase_dim)) + 0.05
        ##
        self.min_max_search_time = min_max_selection_eval_time
        ##
        self.search_mode = False
        self.search_mode_timer = 0
        self.valid_error_model = False

    @classmethod
    def model_smserrors(cls, d_y_predicted, d_grounds, target_weight):
        """

        @param d_y_predicted: (model_n, data_n, sens_n, phase_n)
        @param d_grounds:  (sens_n, phase_n)
        @return: (model_n, data_n, sens_n, phase_n)
        """
        return [np.sqrt(
            np.sum(np.square((d_grounds - d_y_predicted[i])) * target_weight, axis=(1, 2)))
            for i in range(len(d_y_predicted))]

    def _restart_errors(self):
        for i in range(len(self._models)):
            self._models_logodds[i] = 0.
            self._model_signed_errors[i] = self.error_models_mean[i]

    @classmethod
    def surprise(cls, error, error_model_mean, error_model_std, theta):
        return error > error_model_mean + theta * cls._std_nrm(error_model_std)

    def update_error_models(self, errors: List[np.ndarray], alphas: np.ndarray):
        self.error_models_mean = [self.error_models_mean[i] * (1 - alphas[i]) + np.mean(err) * alphas[i]
                                  for i, err in enumerate(errors)]
        self.error_models_stdv = [self.error_models_stdv[i] * (1 - alphas[i]) + np.std(err) * alphas[i]
                                  for i, err in enumerate(errors)]

    def _update_model_log_odds(self, log_odds):
        for i in range(len(self._models)):
            self._models_logodds[i] = log_odds[i] * self._models_confidence_lr + \
                                      self._models_logodds[i] * (1 - self._models_confidence_lr)
        self.log_value(self.R_MODEL_LOGODDS, np.asarray(self._models_logodds))

    def _update_model_signed_errors(self, signed_errors):
        for i in range(len(self._models)):
            self._model_signed_errors[i] = signed_errors[i] * self._models_confidence_lr + \
                                           self._model_signed_errors[i] * (1 - self._models_confidence_lr)

    def direction_diff(self,
                       dy_grounds: np.ndarray,
                       dy_prediction: np.ndarray,
                       model_prediction_deviation: np.ndarray,
                       zero_neighbourhood_epsilon: float,
                       d_yu_variance: np.ndarray
                       ):
        _zero_epsilon = self._std_nrm(
            np.sqrt(self._odometry_sum_transformation(d_yu_variance))) * zero_neighbourhood_epsilon
        ## COMPARISON WITH ZERO-MODEL
        dy_pb_is_zero = np.zeros(dy_prediction.shape)
        dy_pb_is_zero[(_zero_epsilon - np.abs(dy_prediction)) > 0] = 1.
        ## Outside (motion) comparison
        dy_pb = dy_prediction * (1 - dy_pb_is_zero)
        d_y_pred_hist = self.d_y_hist(
            d_y=dy_pb, d_y_std=model_prediction_deviation,
            zero_epsilon=_zero_epsilon)
        d_ground_dirac = self.d_y_dirac(dy_grounds, zero_epsilon=_zero_epsilon)
        # getting (neg) entropies
        best_model_entropy_out = self.d_y_dirac_hist_entropy(d_ground_dirac, d_y_pred_hist)
        self.log_value(self.R_BEST_NEGENTROPY, best_model_entropy_out[-1])
        ## Inside (non-motion) comparison
        """
        d_y_pred_hist -> (f-, f0, f+) = F
        d_ground_dirac -> (y-, y0, y+) =  Y
        diff = F - Y
        e = ||diff * weights||
        N(e; mu,sigm) <- Error model
        e(t) > mu + theta * sigm => "learn a new model"
        """
        return np.argmax(d_y_pred_hist, axis=3) - np.argmax(d_ground_dirac, axis=3)

    def push_suggestion(self, suggestion: int) -> int:
        return suggestion

    def two_stage_comparison(self,
                             dy_grounds: np.ndarray,
                             dy_predictions: List[np.ndarray],
                             model_prediction_deviations: List[np.ndarray]):

        _target_weight = self.current_target_parameter.weight
        target_weight = _target_weight / np.sum(_target_weight)
        is_best_model_good = True
        selected_model = self._current_model_selection
        # norm error: e = ||diff_pred||
        errors = self.model_smserrors(d_y_predicted=dy_predictions, d_grounds=dy_grounds, target_weight=target_weight)
        dir_diffs = [self.direction_diff(
            dy_grounds=dy_grounds, dy_prediction=dy_predictions[i],
            model_prediction_deviation=model_prediction_deviations[i],
            zero_neighbourhood_epsilon=self._zero_neighbourhood_epsilon,
            d_yu_variance=self._models[i].d_yu_variance
        )
            for i in range(len(dy_predictions))]

        signed_errors = [np.sqrt(np.sum(np.square(dir_diff) * target_weight[None, :, :], axis=(1, 2)))
                         for dir_diff in dir_diffs]

        self._update_model_log_odds([error[-1] for error in errors])  # -> self._models_logodds[i]
        self._update_model_signed_errors([error[-1] for error in signed_errors])  # -> self._model_signed_errors[i]

        if self._gait_counter == self._gait_buffer_size:
            alphas = np.zeros((len(self._models),)) + 1
            self.update_error_models(signed_errors, alphas)
            self.valid_error_model = True

        # error model: E = {<e>, std(e)}
        # surprise(e, E): e > E_mean + theta * E_std
        """Is surprising"""
        # IF surprise(e^*, E^*) THEN activate searching mode: ts=0
        if self.surprise(self._model_signed_errors[selected_model],
                         error_model_mean=self.error_models_mean[selected_model],
                         error_model_std=self.error_models_stdv[selected_model],
                         theta=self._min_confident_elements_rate) and not self.search_mode and self.valid_error_model:
            self.search_mode = True
            self.search_mode_timer = 0
            self._restart_errors()

        """Searching mode"""
        if self.search_mode:
            # IF ts > tmin and any i-model not surprise(e^i,E^i) THEN * = best i (quit searching mode)
            is_surprises = [self.surprise(self._model_signed_errors[i], error_model_mean=self.error_models_mean[i],
                                          error_model_std=self.error_models_stdv[i],
                                          theta=self._min_confident_elements_rate)
                            for i in range(len(self._models))]
            if self.search_mode_timer > self.min_max_search_time[0] and not all(is_surprises):
                selected_model = np.argmin(self._models_logodds)
                self.search_mode = False
            # ELIF ts > tmax: ZERO_MODEL_ID (quit searching mode)
            elif self.search_mode_timer > self.min_max_search_time[1]:
                is_best_model_good = False
                self.search_mode = False
        else:
            # alphas = np.zeros((len(self._models),))
            # alphas[selected_model] = self._models_confidence_lr / 2
            # self.update_error_models(errors, alphas)
            pass
        wm_uncertainty = (self._model_signed_errors[selected_model] - self.error_models_mean[
            selected_model]) / self._std_nrm(self.error_models_stdv[selected_model])
        self.log_value(self.R_FIRST_STAGE_SCORE,
                       -np.asarray(self._models_logodds))  # taking negative to unify the data postprocess
        self.log_value(self.R_SECOND_STAGE_SCORE, - wm_uncertainty)
        self.log_value(self.R_CONFIDENCE_AGGREGATE, - wm_uncertainty * np.ones(dy_grounds[0].shape))  # emb

        ## DECISION
        self.search_mode_timer += 1
        if is_best_model_good:
            return selected_model
        else:
            return self.ZERO_MODEL_ID
