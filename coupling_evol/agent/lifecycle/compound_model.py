import coupling_evol.agent.components.internal_model.forward_model as FM
import numpy as np
from typing import List


def one_hot_matrix(a, ncols):
    out = np.zeros((ncols, a.size))
    out[a.ravel(), np.arange(a.size)] = 1
    out.shape = (ncols,)+ a.shape
    return out


class CompoundModel(object):
    def __init__(self, models: List[FM.MultiPhaseModel], model_modality_phase_weights: np.ndarray, use_best_model_u_stats=True):
        """
        Assumes use of linear regressors
        @param models:
        @type models:
        @param model_modality_phase_weights: (model_n, sens_dim, ph_dim)
        @type model_modality_phase_weights:
        @return:
        @rtype:
        """
        self._models = models
        self._phase_n = models[0].phase_n
        self._u_dim = models[0].u_dim
        self._y_dim = models[0].y_dim

        self._model_modality_phase_weights = model_modality_phase_weights
        self._model_weights = np.mean(model_modality_phase_weights, axis=(1,2))
        # self._model_logodds = np.zeros((model_logodds.shape)) - 100
        # self._model_logodds[-1, :, :] = -1
        self.best_model_id = np.argmax(np.sum(model_modality_phase_weights, axis=(1, 2)))

        if use_best_model_u_stats:
            self._best_model = models[int(self.best_model_id)]
            self._u_mean = self._best_model.u_mean
            self._u_std = self._best_model.u_std
        else:
            self._u_mean = np.sum(np.asarray([m.u_mean for m in models]) * self._model_weights[:, None, None], axis=0)
            self._u_std = np.sqrt(
                np.sum(np.square(np.asarray([m.u_std for m in models])) * self._model_weights[:, None, None], axis=0))
        
        ##
        # self._normed_weights = self._softmax_logodds(self._model_modality_phase_weights)
        # self._normed_weights = one_hot_matrix(np.argmax(self._model_modality_phase_weights, axis=0), self._normed_weights.shape[0])
            
        # tmp = np.power(self._normed_weights, 100)
        # self._normed_weights = tmp/ np.sum(tmp, axis=0)

        self._y_mean = self._combine_sensory_embeddings(self._model_modality_phase_weights, np.asarray([m.y_mean for m in models]))
        self._d_yu_variance = self._combine_sensory_embeddings(self._model_modality_phase_weights, np.asarray([m.d_yu_variance for m in models]))

        # since we work with linear models we can precompute derivative
        self._precomputed_derivatives = np.asarray([
            self._derivative_gait(
                models, self._model_modality_phase_weights, np.zeros((self._u_dim, self._phase_n)),
                p)
            for p in range(self._phase_n)])


    def get_model_modality_phase_weights(self):
        return self._model_modality_phase_weights

    @staticmethod
    def _softmax_logodds(logodds: np.ndarray) -> np.ndarray:
        """
        @param logodds: (model_n, sens_dim, ph_dim)
        @return: (model_n, sens_dim, ph_dim)
        """
        exp_logodds = np.exp(logodds)
        return exp_logodds / np.minimum(np.sum(exp_logodds, axis=0), 0.00001)

    @staticmethod
    def _combine_sensory_embeddings(model_modality_phase_weights, sensory_embeddings):
        """
        @param model_modality_phase_weights: (model_n, sens_dim, ph_dim) or (model_n, data_n, sens_dim, ph_dim)
        @param sensory_embeddings: (model_n, sens_dim, ph_dim)
        @return: (sens_dim, ph_dim) or (data_n, sens_dim, ph_dim)
        """
        if sensory_embeddings.ndim == 3:
            return np.sum(model_modality_phase_weights * sensory_embeddings, axis=0)
        elif sensory_embeddings.ndim == 4:
            return np.sum(model_modality_phase_weights[:, None, :, :] * sensory_embeddings, axis=0)

    def combine_sensory_embeddings(self, sensory_embeddings):
        return self._combine_sensory_embeddings(self._model_modality_phase_weights, sensory_embeddings)

    @staticmethod
    def _combine_embedded_derivatives(normed_weights, sensory_embeddings):
        """
        @param normed_weights: (model_n, sens_dim, ph_dim)
        @param sensory_embeddings: (model_n, sens_dim, ph_dim)
        @return: (sens_dim, ph_dim)
        """
        return np.sum(normed_weights * sensory_embeddings, axis=0)

    @staticmethod
    def _derivative_gait(
            models: List[FM.MultiPhaseModel], normed_weights, gait_embedding: np.ndarray, sensory_phase: int):
        """

        @param models:
        @param normed_weights: (model_n, sens_dim, ph_dim)
        @param gait:
        @param phase:
        @return:
        """
        dFm = np.asarray([m.derivative_gait(gait_embedding, sensory_phase) for m in models])
        return np.einsum("ns,nsmo->smo", normed_weights[:, :, sensory_phase], dFm)

    def derivative_gait(self, gait, phase) -> np.ndarray:
        return self._precomputed_derivatives[phase, :, :, :]

    def predict_gait_response(self, u_embeds_opt) -> np.ndarray:
        return self._combine_sensory_embeddings(
            self._model_modality_phase_weights,
            np.asarray([m.predict_gait_response(u_embeds_opt) for m in self._models]))

    @property
    def u_mean(self) -> np.ndarray:
        return self._u_mean

    @property
    def u_std(self) -> np.ndarray:
        return self._u_std

    @property
    def phase_n(self) -> np.ndarray:
        return self._phase_n

    @property
    def u_dim(self) -> np.ndarray:
        return self._u_dim

    @property
    def y_dim(self) -> np.ndarray:
        return self._y_dim

    @property
    def y_mean(self) -> np.ndarray:
        return self._y_mean

    @property
    def d_yu_variance(self) -> np.ndarray:
        return self._d_yu_variance


if __name__ == '__main__':
    print("ff")
    x = np.asarray([[0,1,0],[0,0,0]])
    print(one_hot_matrix(x, 2)[0])