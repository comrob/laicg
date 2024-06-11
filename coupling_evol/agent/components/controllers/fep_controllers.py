from coupling_evol.engine import embedded_control as C
import numpy as np
from coupling_evol.agent.components.internal_model.forward_model import MultiPhaseModel
from typing import Dict
from enum import Enum

import coupling_evol.data_process.inprocess.recorder as rlog
from coupling_evol.engine.common import RecordNamespace
from abc import ABC

REC = rlog.get_recorder(prefix=RecordNamespace.LIFE_CYCLE.key, postfix=C.EmbeddingController.CONTROL_PSFX)

ERR_PERFORMANCE = "err_prf"
DELTA_MAGNITUDE = "m_delta_u"
BASE_GAIT = "base_u"
PRIOR_ERROR = "err_p"
LIKELIHOOD_ERROR = "err_l"
LIKELIHOOD_ERROR_VARIANCE = "err_l_v"
TARGET_ERROR_VARIANCE = "err_trg_v"
SENSORY_ESTIMATE = "y_est"
SENSORY_OBSERVATION = "y_val_obs"


class TargetErrorProcessing(Enum):
    ELEMENT_WISE = 0,
    MODE_PHASE_SUM = 1,
    ODOMETRY_SUM_EFFORT_WISE = 2

    @classmethod
    def from_int(cls, val: int):
        if val == 0:
            return cls.ELEMENT_WISE
        elif val == 1:
            return cls.MODE_PHASE_SUM
        elif val == 2:
            return cls.ODOMETRY_SUM_EFFORT_WISE
        else:
            raise NotImplemented("Target process not impl.")


class FepController(ABC, C.EmbeddingController):
    def __init__(self, model: MultiPhaseModel):
        self.model = model


class WaveFepFusionController(FepController):
    """
    Sensory estimate \tilde y update

    \tilde y = \max_x p(x| y, \hat y)
    search such a "x" that maximizes the evidence, where the evidence is provided by
    observation "y" and prediction "\hat y(u)".

    The minimum is searched with gradient descend:

    \dot\tilde y = (y - \tilde y)/S + (\hat y - \tilde y)/\tilde S + (\bar y - \tilde y)/\bar S

    the terms correspond to observation, prediction, and prior respectively.

    Motor \tilde u update

    \tilde u = \max_x p(u| \tilde y = y*)
    search such an "u" that is most likely for estimation being at target y*.

    \dot\tilde u = [(y*-\tilde y)/S*] * [1/\hat S] * [d\hat y(u)/du]

    Variance update
    \dot\hat S = {[(\hat y - \tilde y)/\tilde S]^2 - [1/\tilde S]}/2
    """

    # Numerical stability fix
    VARIANCE_LOWER_BOUND = 1  # IF VAR<this THEN VAR := this
    VARIANCE_UPPER_BOUND = 20  # IF VAR>this THEN VAR := this
    VARIANCE_INVERSE_LIMIT = 15  # IF VAR>this THEN 1/VAR = 0

    #
    PHASES_NAME = "phss"
    AMPLITUDES_NAME = "amps"
    AMPLITUDES_SCALE_NAME = "amps_sc"
    _ERROR_BOUNDARY = 1.2
    JOINT_SYMMETRY_LEFT = [0, 1, 2, 6, 7, 8, 12, 13, 14]
    JOINT_SYMMETRY_RIGHT = [3, 4, 5, 9, 10, 11, 15, 16, 17]

    def __init__(self, model: MultiPhaseModel, prev_control_snap: Dict[str, np.ndarray], gait_learning_rate=0.1,
                 likelihood_variance_learning_rate=0.01, prior_strength=0.001, symmetry_strength=0.,
                 observed_variance=0.1, estimation_prior_variance=10., estimation_learning_rate=0.1,
                 prediction_variance_lower_bound=1., target_error_variance_learning_rate=0.01,
                 prediction_precision_scale=1., target_precision_scale=1.,
                 target_error_processing=TargetErrorProcessing.MODE_PHASE_SUM, sum_amplitude_energy=9,
                 sum_amplitude_energy_strength=0., simplified_gradient_switch=False):
        """
        I try to preserve most of inherited variables with the same meaning,
        but this theoretical model uses two probabilistic queries: estimate query and motor query, where the previous
        theoretical models used only the latter.

        There is one major difference in naming-meaning:
        'likelihood_error' means prediction error; as opposed to performance error. The performance error is now
        computed as an expression of target and estimation scaled by weight. Prediction error is a difference between
        model prediction (based on gait) and the estimate.

        And one minor specification:
        theory gives two priors 'motor prior' and 'estimation prior', the former was used previously and is
        parametrized by 'prior_strength' while the latter is outcome of current model and is parametrized by
        'estimation_prior_variance'.


        Target phase sum mode:
        Target relaxation where we aim for sum over all sensory-mode measurements,
        rather than having phase-wise targets.
        """
        # super().__init__(model, prev_control_snap,
        #                  gait_learning_rate=gait_learning_rate,
        #                  likelihood_variance_learning_rate=likelihood_variance_learning_rate,
        #                  prior_strength=prior_strength,
        #                  symmetry_strength=symmetry_strength,
        #                  )

        # self.target_phase_sum_mode = target_phase_sum_mode
        super().__init__(model)
        self.target_error_processing = target_error_processing
        self.amplitudes_scale = np.zeros((model.u_dim,))
        self.phases = np.zeros((model.u_dim,))
        self.prior_error = np.zeros(self.model.u_mean.shape)
        self.prior_strength = prior_strength
        self.gait_learning_rate = gait_learning_rate
        self.likelihood_variance_learning_rate = likelihood_variance_learning_rate
        self.current_delta_magnitude = 0.
        self.gait = np.zeros((model.u_dim, model.phase_n)) + model.u_mean
        self.likelihood_error = np.zeros(self.model.y_mean.shape)

        ### ALGORITHM INIT
        self.valid_sensory_embedding = np.zeros(model.y_mean.shape) + model.y_mean
        # self.grad_gait = np.zeros((self.model.phase_n, self.model.u_dim, self.model.phase_n))
        self.segment_cnt = -1
        self.gait_cnt = -1
        self.last_motion_phase = -1
        self.starting_motion_phase = -1

        ##
        self.target_error_variance_learning_rate = np.zeros(
            self.model.y_mean.shape) + target_error_variance_learning_rate
        if self.target_error_processing is TargetErrorProcessing.ODOMETRY_SUM_EFFORT_WISE:  # FIXME
            # self.target_error_variance_learning_rate[[0, 3, 4], :] = 0. # velx, rotz and vely are hard objectives.
            self.target_error_variance_learning_rate[[0, 3], :] = 0.  # velx, rotz and vely are hard objectives.

        self.prediction_precision_scale = prediction_precision_scale
        self.target_precision_scale = target_precision_scale

        self.sensory_estimation = np.zeros(model.y_mean.shape) + model.y_mean
        self.sensory_prior = np.zeros(model.y_mean.shape)
        # self.prediction_variance = self.likelihood_error_variance
        self.observed_variance = np.ones(model.y_mean.shape) * observed_variance
        self.prior_variance = np.ones(model.y_mean.shape) * estimation_prior_variance

        self.estimation_learning_rate = estimation_learning_rate
        self.likelihood_error_variance = np.zeros(self.model.y_mean.shape) + 1
        self.prediction_variance_lower_bound = prediction_variance_lower_bound
        ##
        self.target_error_variance = np.zeros(self.model.y_mean.shape) + 1
        if self.target_error_processing is TargetErrorProcessing.ODOMETRY_SUM_EFFORT_WISE:  # FIXME
            # self.target_error_variance[[0, 3, 4], :] = 1/self.VARIANCE_INVERSE_LIMIT # velx, rotz and vely are hard objectives.
            self.target_error_variance[[0, 3],
            :] = 1 / self.VARIANCE_INVERSE_LIMIT  # velx, rotz and vely are hard objectives.

        self.sensory_size = model.y_mean.shape[0] * model.y_mean.shape[1]
        self.sum_amplitude_energy = sum_amplitude_energy
        self.sum_amplitude_energy_strength = sum_amplitude_energy_strength

        ## PID like response regulator
        self.last_target_response = np.zeros(self.model.y_mean.shape)
        self.d_target_response = np.zeros(self.model.y_mean.shape)
        self.i_target_response = np.zeros(self.model.y_mean.shape)
        # if TARGET_ERROR_VARIANCE in prev_control_snap:
        # self.target_error_variance = prev_control_snap[TARGET_ERROR_VARIANCE]

        # if LIKELIHOOD_ERROR_VARIANCE in prev_control_snap:
        # self.likelihood_error_variance = prev_control_snap[LIKELIHOOD_ERROR_VARIANCE]
        self.error_boundary = 100

        if self.AMPLITUDES_NAME in prev_control_snap:
            # self.amplitudes = prev_control_snap[self.AMPLITUDES_NAME]
            self.amplitudes_scale = prev_control_snap[self.AMPLITUDES_SCALE_NAME]
        if self.PHASES_NAME in prev_control_snap:
            self.phases = prev_control_snap[self.PHASES_NAME]
        if LIKELIHOOD_ERROR in prev_control_snap:
            self.error_boundary = np.mean(np.square(prev_control_snap[LIKELIHOOD_ERROR])) * self._ERROR_BOUNDARY
        self.phase_centers = np.asarray([2 * np.pi * np.arange(0, model.phase_n) / model.phase_n] * model.u_dim)
        self.symmetry_strength = symmetry_strength


        if simplified_gradient_switch:
            self._gradient_step = self._gradient_step_simplified

    def get_history(self) -> C.List[C.Dict[str, np.ndarray]]:
        # internal state updates
        return [{
            self.PHASES_NAME: np.zeros(self.phases.shape) + self.phases,
            self.AMPLITUDES_NAME: self.amplitude(self.amplitudes_scale),
            self.AMPLITUDES_SCALE_NAME: np.zeros(self.amplitudes_scale.shape) + self.amplitudes_scale,
            ERR_PERFORMANCE: float(self.performance_error),
            DELTA_MAGNITUDE: float(self.last_delta_magnitude),
            PRIOR_ERROR: np.zeros(self.prior_error.shape) + self.prior_error,
            LIKELIHOOD_ERROR: np.zeros(self.likelihood_error.shape) + self.likelihood_error,
            LIKELIHOOD_ERROR_VARIANCE: np.zeros(self.likelihood_error.shape) + self.likelihood_error_variance,
            self.DELTA_GAIT: np.zeros(self.delta_gait.shape) + self.delta_gait,
            SENSORY_ESTIMATE: np.zeros(self.sensory_estimation.shape) + self.sensory_estimation,
            SENSORY_OBSERVATION: np.zeros(self.valid_sensory_embedding.shape) + self.valid_sensory_embedding,
            TARGET_ERROR_VARIANCE: np.zeros(self.target_error_variance.shape) + self.target_error_variance,
            BASE_GAIT: np.zeros(self.delta_gait.shape) + self.gait
        }]

    @staticmethod
    def amplitude(amplitudes_scale):
        return np.tanh(amplitudes_scale)

    @staticmethod
    def _error_variance_gradient(likelihood_error, likelihood_variance):
        return (np.square(likelihood_error) - 1 / likelihood_variance) / 2

    @classmethod
    def _var_inv(cls, var):
        inv = 1 / var
        inv[var > cls.VARIANCE_INVERSE_LIMIT] = 0.
        return inv

    @classmethod
    def _wave_to_gait(cls, amplitudes_scale, phases, segment_centers):
        return cls.amplitude(amplitudes_scale)[:, None] * np.sin(segment_centers + phases[:, None])

    @staticmethod
    def _wave_gradients(amplitudes_scale, phases, segment_centers, gait_gradient):
        tnh_amp_sc = np.tanh(amplitudes_scale)
        d_tnh_amp_sc = 1 - np.square(tnh_amp_sc)
        _d_amp = d_tnh_amp_sc[:, None] * np.sin(segment_centers + phases[:, None])
        _d_phs = np.cos(segment_centers + phases[:, None]) * np.sign(tnh_amp_sc[:, None])
        d_amp_sc = np.mean(gait_gradient * _d_amp, axis=1)
        d_phs = np.mean(gait_gradient * _d_phs, axis=1)
        return d_amp_sc, d_phs

    @staticmethod
    def _symmetry_regularization(amplitudes_scale, symm_one, symm_two):
        """
        E = (symm_one - symm_two)^2
        dE/dsymm_one = 2(symm_one - symm_two)
        @param amplitudes_scale:
        @type amplitudes_scale:
        @return:
        @rtype:
        """
        d_amps = np.zeros((amplitudes_scale.shape[0],))
        d_amps[symm_one] = (amplitudes_scale[symm_one] - amplitudes_scale[symm_two])
        d_amps[symm_two] = (amplitudes_scale[symm_two] - amplitudes_scale[symm_one])
        return d_amps

    def _var_clp(self, var):
        return np.clip(var, a_min=self.prediction_variance_lower_bound, a_max=self.VARIANCE_UPPER_BOUND)

    @staticmethod
    def _estimate_gradient(y_estimate, y_observation, y_prediction, y_prior,
                           C_observation, C_prediction, C_prior
                           ):
        """
        Confidence is inverse variance: C = 1/S .
        @return:
        @rtype:
        """
        return (y_observation - y_estimate) * C_observation + \
            (y_prediction - y_estimate) * C_prediction + \
            (y_prior - y_estimate) * C_prior

    def predict_gait_sensory_effect(self, gait):
        return self.model.predict_gait_response(np.asarray([gait]))[0, :, :]

    def _gait_gradient_phase_sum_target(self, gait, target_response, C_prediction):
        """
        For phase-motor mk:
        \dot u_mp = \Sum_n^(sensors) [target_response_n * {\Sum_f^(granularity) dy(u)_nf/du_mp}]
        @param gait: (M, P), motor dim and (motor) granularity
        @param target_response: (N, ) sensory dim
        @param C_prediction: (N, F), motor dim and (sensory) granularity
        @return:
        @rtype:
        """
        dF = np.asarray([self.model.derivative_gait(gait, phase) for phase in range(self.model.phase_n)])
        return np.einsum("n,fnmp,nf->mp", target_response, dF, C_prediction)

    def _gait_gradient_phase_wise_target(self, gait, target_response, C_prediction):
        """
        For phase-motor mk:
        \dot u_mp = \Sum_nf^(phase-sensors) [target_response_nf * dy(u)_nf/du_mp]
        @param gait: (M, P), motor dim and (motor) granularity
        @param target_response: (N, F) sensory dim and (sensory) granularity
        @param C_prediction: (N, F), motor dim and (sensory) granularity
        @return:
        @rtype:
        """
        dF = np.asarray([self.model.derivative_gait(gait, phase) for phase in range(self.model.phase_n)])
        return np.einsum("nf,fnmp->mp", target_response * C_prediction, dF) / self.sensory_size

    def _gait_gradient_odometry_sum_effort_wise_target(self, gait, target_response, C_prediction):
        dF = np.asarray([self.model.derivative_gait(gait, phase) for phase in range(self.model.phase_n)])
        el_wise = np.einsum("nf,fnmp->nmp", target_response * C_prediction, dF)
        el_sum = np.einsum("n,fnmp,nf->nmp", target_response[:, 0], dF, C_prediction)
        return (np.sum(el_sum[:5, :, :], axis=0) + np.sum(el_wise[5:, :, :], axis=0)) / self.sensory_size

    def _amplitude_gradient_regularization(self, amp):
        return -amp

    def _amplitude_energy_regularization(self, amp, amp_energy):
        energy = np.square(amp)
        return amp * (amp_energy - np.sum(energy))

    def update_PID(self, target_response):
        self.d_target_response += (target_response - self.last_target_response) * 0.001
        self.i_target_response += target_response * 0.001
        self.last_target_response = target_response

    def get_ID_sum(self):
        return self.d_target_response + self.i_target_response

    def _gradient_step(self, diff):
        # FIRST: we resolve some expressions
        C_predicton = self._var_inv(self.likelihood_error_variance)
        C_target = self._var_inv(self.target_error_variance)
        y_prediction = self.predict_gait_sensory_effect(self.gait)
        """likelihood_error (prediction response) is not integrated but an expression of prediction and estimation"""
        self.likelihood_error = (y_prediction - self.sensory_estimation) * C_predicton
        target_response = diff * C_target
        # SECOND: we get all grads
        """y is now estimated by integrated multiple sources (thus previous error integrations shouldn't be needed)"""
        d_y_est = self._estimate_gradient(y_estimate=self.sensory_estimation,
                                          y_observation=self.valid_sensory_embedding,
                                          y_prediction=y_prediction,
                                          y_prior=self.sensory_prior,
                                          C_observation=1 / self.observed_variance,
                                          C_prediction=C_predicton,
                                          C_prior=1 / self.prior_variance
                                          )

        """err_likelihood_var (prediction variance)"""
        d_err_likelihood_var = self._error_variance_gradient(
            self.likelihood_error * self.prediction_precision_scale, self.likelihood_error_variance)
        """target_error_variance (target variance)"""
        d_target_error_variance = self._error_variance_gradient(
            target_response * self.target_precision_scale, self.target_error_variance)

        """
        The diff (target response) is now scaled with prediction confidence.
        Prior should be part of regularization as it is not part of gradient anymore.
        """
        if self.target_error_processing is TargetErrorProcessing.MODE_PHASE_SUM:
            d_gait = self._gait_gradient_phase_sum_target(self.gait, target_response[:, 0], C_predicton)
        elif self.target_error_processing is TargetErrorProcessing.ODOMETRY_SUM_EFFORT_WISE:
            d_gait = self._gait_gradient_odometry_sum_effort_wise_target(self.gait, target_response, C_predicton)
        else:
            d_gait = self._gait_gradient_phase_wise_target(self.gait, target_response, C_predicton)

        """
        Prior regularizaiton: added in an adhoc way.
        """
        d_gait_prior = (self.gait - self.model.u_mean)
        d_gait = d_gait + d_gait_prior * self.prior_strength
        # Which is correct because the "d/dyref" negation happens at the final layer
        """Gait encoding and regularizations"""
        d_amp_sc, d_phs = self._wave_gradients(self.amplitudes_scale, self.phases, self.phase_centers, d_gait)
        if self.symmetry_strength > 0. and self.model.u_dim == 18:
            d_amp_symm_reg = self._symmetry_regularization(self.amplitudes_scale,
                                                           symm_one=self.JOINT_SYMMETRY_LEFT,
                                                           symm_two=self.JOINT_SYMMETRY_RIGHT
                                                           )
        else:
            d_amp_symm_reg = 0.
        d_amp_zero_reg = self._amplitude_energy_regularization(self.amplitudes_scale,
                                                               amp_energy=self.sum_amplitude_energy)
        d_amp_reg = -d_amp_zero_reg * self.sum_amplitude_energy_strength + d_amp_symm_reg * self.symmetry_strength

        # THIRD: the gradient step
        self.sensory_estimation += d_y_est * self.estimation_learning_rate
        self.amplitudes_scale -= (d_amp_sc + d_amp_reg) * self.gait_learning_rate
        self.phases -= d_phs * self.gait_learning_rate
        # dstd = (error - 1)/std -> meaning if above 1 std (var) will grow but the growth will slow down
        # however if error is lesser than one then the variance decays faster the lower it is.
        # Thus the variance should be lower-bound capped
        self.likelihood_error_variance += np.tanh(d_err_likelihood_var) * self.likelihood_variance_learning_rate
        self.likelihood_error_variance = self._var_clp(self.likelihood_error_variance)
        ##
        self.target_error_variance += np.tanh(d_target_error_variance) * self.target_error_variance_learning_rate
        self.target_error_variance = self._var_clp(self.target_error_variance)
        # FOURTH: gait expression resolving
        self.gait = self._wave_to_gait(amplitudes_scale=self.amplitudes_scale, phases=self.phases,
                                       segment_centers=self.phase_centers)

    def _gradient_step_simplified(self, diff):
        """
        The new gradient step is simplified to the following:
        1. The sensory estimation is purely taken from the sensory observation.
        2. The target response has PID like regulation.
        3. No variance update.
        4. The gait is updated by the target response and the sensory estimation, without prediction variance.
        """
        # FIRST: we resolve some expressions
        y_prediction = self.predict_gait_sensory_effect(self.gait)
        """likelihood_error (prediction response) is not integrated but an expression of prediction and estimation"""
        self.likelihood_error = (y_prediction - self.sensory_estimation)
        _target_response = diff
        target_response = _target_response + self.get_ID_sum()
        # SECOND: we get all grads
        """y is now estimated by integrated multiple sources (thus previous error integrations shouldn't be needed)"""
        d_y_est = self._estimate_gradient(y_estimate=self.sensory_estimation,
                                          y_observation=self.valid_sensory_embedding,
                                          y_prediction=y_prediction,
                                          y_prior=self.sensory_prior,
                                          C_observation=1 / self.observed_variance,
                                          C_prediction=1 / self.prediction_variance_lower_bound,
                                          C_prior=1 / self.prior_variance
                                          )
        """
        The diff (target response) is now scaled with prediction confidence.
        Prior should be part of regularization as it is not part of gradient anymore.
        """
        constant_pred = np.ones_like(diff)
        if self.target_error_processing is TargetErrorProcessing.MODE_PHASE_SUM:
            d_gait = self._gait_gradient_phase_sum_target(self.gait, target_response[:, 0], constant_pred)
        elif self.target_error_processing is TargetErrorProcessing.ODOMETRY_SUM_EFFORT_WISE:
            d_gait = self._gait_gradient_odometry_sum_effort_wise_target(self.gait, target_response, constant_pred)
        else:
            d_gait = self._gait_gradient_phase_wise_target(self.gait, target_response, constant_pred)
        """
        Prior regularizaiton: added in an adhoc way.
        """
        d_gait_prior = (self.gait - self.model.u_mean)
        d_gait = d_gait + d_gait_prior * self.prior_strength
        # Which is correct because the "d/dyref" negation happens at the final layer
        """Gait encoding and regularizations"""
        d_amp_sc, d_phs = self._wave_gradients(self.amplitudes_scale, self.phases, self.phase_centers, d_gait)
        if self.symmetry_strength > 0. and self.model.u_dim == 18:
            d_amp_symm_reg = self._symmetry_regularization(self.amplitudes_scale,
                                                           symm_one=self.JOINT_SYMMETRY_LEFT,
                                                           symm_two=self.JOINT_SYMMETRY_RIGHT
                                                           )
        else:
            d_amp_symm_reg = 0.
        d_amp_zero_reg = self._amplitude_energy_regularization(self.amplitudes_scale,
                                                               amp_energy=self.sum_amplitude_energy)
        d_amp_reg = -d_amp_zero_reg * self.sum_amplitude_energy_strength + d_amp_symm_reg * self.symmetry_strength

        # THIRD: the gradient step
        self.update_PID(_target_response)
        self.sensory_estimation += d_y_est * self.estimation_learning_rate
        self.amplitudes_scale -= (d_amp_sc + d_amp_reg) * self.gait_learning_rate
        self.phases -= d_phs * self.gait_learning_rate
        # FOURTH: gait expression resolving
        self.gait = self._wave_to_gait(amplitudes_scale=self.amplitudes_scale, phases=self.phases,
                                       segment_centers=self.phase_centers)

    def __call__(self, sensory_embedding: np.ndarray, target_parameter: C.EmbeddedTargetParameter, motion_phase):
        phn = np.argmax(motion_phase)
        if self.last_motion_phase == -1:
            self.starting_motion_phase = phn
        if self.last_motion_phase != phn:
            # ---DONT TOUCH
            self.valid_sensory_embedding[:, self.last_motion_phase] = sensory_embedding[:, self.last_motion_phase]
            self.last_motion_phase = phn
            self.segment_cnt += 1
            if self.segment_cnt % self.model.phase_n == 0:
                self.gait_cnt += 1
            # -------------
            if self.target_error_processing is TargetErrorProcessing.MODE_PHASE_SUM:
                diff = target_parameter.embedding_phase_sum_difference(self.sensory_estimation)
                diff = np.zeros(self.sensory_estimation.shape) + diff[:, None]

            elif self.target_error_processing is TargetErrorProcessing.ODOMETRY_SUM_EFFORT_WISE:
                diff_sum = target_parameter.embedding_phase_sum_difference(self.sensory_estimation)
                diff_wise = target_parameter.embedding_difference(self.sensory_estimation)
                diff = np.zeros(self.sensory_estimation.shape)
                diff[:5, :] = diff_sum[:5, None]
                diff[5:, :] = diff_wise[5:, :]

            else:
                diff = target_parameter.embedding_difference(self.sensory_estimation)
            if self.gait_cnt > 1:  # each new segment after one gait
                self._gradient_step(diff=diff)
            if phn == self.starting_motion_phase:
                self.performance_error = np.mean(np.square(diff + self.get_ID_sum()))
                self.last_delta_magnitude = self.current_delta_magnitude
                self.current_delta_magnitude = np.linalg.norm(self.gait - self.model.u_mean)
                self.delta_gait = self.gait - self.model.u_mean

        REC(self.PHASES_NAME, np.zeros(self.phases.shape) + self.phases)
        REC(self.AMPLITUDES_NAME, self.amplitude(self.amplitudes_scale))
        REC(self.AMPLITUDES_SCALE_NAME, np.zeros(self.amplitudes_scale.shape) + self.amplitudes_scale)
        REC(ERR_PERFORMANCE, float(self.performance_error))
        REC(DELTA_MAGNITUDE, float(self.last_delta_magnitude))
        REC(PRIOR_ERROR, np.zeros(self.prior_error.shape) + self.prior_error)
        REC(LIKELIHOOD_ERROR, np.zeros(self.likelihood_error.shape) + self.likelihood_error)
        REC(LIKELIHOOD_ERROR_VARIANCE, np.zeros(self.likelihood_error.shape) + self.likelihood_error_variance)
        REC(self.DELTA_GAIT, np.zeros(self.delta_gait.shape) + self.delta_gait)
        REC(SENSORY_ESTIMATE, np.zeros(self.sensory_estimation.shape) + self.sensory_estimation)
        REC(SENSORY_OBSERVATION, np.zeros(self.valid_sensory_embedding.shape) + self.valid_sensory_embedding)
        REC(TARGET_ERROR_VARIANCE, np.zeros(self.target_error_variance.shape) + self.target_error_variance)
        REC(BASE_GAIT, np.zeros(self.delta_gait.shape) + self.gait)

        ##
        return self.gait


class MultiFepControllerContainer(C.EmbeddingController):

    def __init__(self, controllers: C.List[FepController]):
        self.controllers = controllers
        self.model_motor_phase_weights = np.zeros(
            (len(controllers), controllers[0].model.u_mean.shape[0], controllers[0].model.phase_n))

    def set_model_motor_phase_weights(self, weights: np.ndarray):
        self.model_motor_phase_weights = weights

    def get_best_model_weight_id(self):
        return np.argmax(np.sum(self.model_motor_phase_weights, axis=(1, 2)))

    def __call__(self, sensory_embedding: np.ndarray, target_parameter: C.EmbeddedTargetParameter, motion_phase):
        controls = [c(sensory_embedding, target_parameter, motion_phase) for c in self.controllers]
        combined_weighted_control = np.sum(np.asarray(controls) * self.model_motor_phase_weights, axis=0)
        return combined_weighted_control

    def get_history(self) -> C.List[C.Dict[str, np.ndarray]]:
        histories = [c.get_history() for c in self.controllers]
        best_model_id = self.get_best_model_weight_id()
        return histories[best_model_id]
