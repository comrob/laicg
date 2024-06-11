import os
import numpy as np
import yaml
import time
from coupling_evol.agent.components.internal_model import regressors as R
import coupling_evol.agent.components.embedding.cpg_rbf_embedding as E

MODEL_LOADERS = dict([(cls.__name__, cls.load) for cls in [
    R.common.JustMean,
    R.sklearn_regressors.GaussianProcessRbf,
    R.sklearn_regressors.GaussianProcessDot,
    R.sklearn_regressors.StdNormedPerceptronRegressor,
    R.sklearn_regressors.StdNormedLinearRegressor,
    R.sklearn_regressors.StdNormedRidgeRegressor,
]])


class MultiPhaseModel:
    NUMERICAL_EPS = 0.000001

    def __init__(self, model_builder, degenerate_model_builder, y_dimension, u_dimension, phase_number,
                 degenerate_datasize_threshold=10, transgait_window=0):
        self.models = []
        self.model_builder = model_builder
        self.degenerate_model_builder = degenerate_model_builder
        self.degenerate_threshold = degenerate_datasize_threshold
        self.y_dim = y_dimension
        self.u_dim = u_dimension
        self.phase_n = phase_number
        # u_embeds stats
        self.u_mean = np.zeros((u_dimension, phase_number))
        self.u_std = np.ones((u_dimension, phase_number))
        self._u_mean_flat = np.zeros((u_dimension * phase_number))
        self._u_std_flat = np.ones((u_dimension * phase_number))
        # y_embeds stats
        self.y_mean = np.zeros((y_dimension, phase_number))
        self.y_std = np.ones((y_dimension, phase_number))
        self.yu_variance = np.ones((y_dimension, phase_number))
        self.d_yu_variance = np.ones((y_dimension, phase_number))
        # embedding window
        self.transgait_window = transgait_window
        ## norm derivative
        self._derivative_norm = np.ones((y_dimension, u_dimension, phase_number, phase_number))

    def get_generative_sample(self, number):
        """

        @return: p(y|M)
        """
        sample = np.random.randn(number * self.u_dim * self.phase_n).reshape((number, self.u_dim, self.phase_n))
        return self.predict_gait_response(flatten(sample * self.u_std[None, :, :] + self.u_mean[None, :, :]))

    def norm_u(self, u_trans_embeds):
        return (u_trans_embeds - self.u_mean) / self.u_std

    def norm_flattened_u(self, u_simple_flattened_embeds):
        return (u_simple_flattened_embeds - self._u_mean_flat) / self._u_std_flat

    def denorm_u(self, u_trans_embeds):
        return u_trans_embeds * self.u_std + self.u_mean

    def norm_y(self, y_vecs, phase: int):
        return (y_vecs - self.y_mean[:, phase]) / self.y_std[:, phase]

    def denorm_y(self, y_vecs, phase: int):
        return y_vecs * self.y_std[:, phase] + self.y_mean[:, phase]

    def u_prepare(self, u_trans_embeddings):
        return flatten_trans_embedding(self.norm_u(u_trans_embeddings))

    def phase_predict(self, u_prepared, phase: int):
        return self.denorm_y(self.models[phase].predict(u_prepared), phase)

    def _phase_derivative(self, u_prepared, sensory_phase: int):
        """
        F(u, phi) = Sum[|phi=ph|] denorm^y_ph(f_ph(norm^u(u)))
        we care just for particular sensory_phase derivative
        dF/du|phi=ph = ddenorm^y_ph/df_ph(norm^u(u)) * df_ph/d(norm^u(u)) *  dnorm^u(u)/du ... chain rule
        dF/du|phi=ph =  df_ph/d(norm^u(u)) * std^y_ph/std^u

        - the "norm^u(u)" is done in the u_prepared
        - the "df_ph/d(norm^u(u))" is done by model[sensory_phase]
        - the std^y_ph/std^u is precomputed from stats as _derivative_norm
        @param u_prepared: flattened embedding (transgait_window * u_dim * granularity)
        @param sensory_phase: sensory phase for which the "gait responsibility" is computed
        @return: derivative (y_dim, u_dim, motor_dim)
        """
        df_ph = de_flatten_trans_embedding(
            self.models[sensory_phase].derivative(u_prepared),
            trans_gait_window_size=self.transgait_window,
            dimension=self.u_dim,
            granularity=self.phase_n
        )
        # FIXME one day when I will care about transgaits again this might need fixing
        return df_ph[:, 0, :, :] * self._derivative_norm[:, :, :, sensory_phase]

    def derivative_gait(self, gait_embedding: np.ndarray, sensory_phase: int):
        """

        @param gait_embedding: (u_dim, granularity)
        @param sensory_phase:
        @return: (y_dim, u_dim, motor_dim)
        """
        flt = self.norm_flattened_u(flatten(gait_embedding))
        u_embeds_flat = np.concatenate([flt] * self.transgait_window)
        return self._phase_derivative(u_embeds_flat, sensory_phase)

    def _fit_prior_stats(self, u_trans_embeds, y_vecs, phases):
        self.u_mean = np.mean(u_trans_embeds, axis=(0, 1))
        # self.u_std = np.maximum(np.std(u_trans_embeds, axis=(0, 1)), self.NUMERICAL_EPS)
        self.u_std = np.maximum(np.std(u_trans_embeds, axis=(0, 1)), self.NUMERICAL_EPS)
        self._u_mean_flat = flatten(self.u_mean)
        self._u_std_flat = flatten(self.u_std)
        for ph in range(self.phase_n):
            ph_sel = (phases[:, ph] == 1)
            y = y_vecs[ph_sel]
            ##
            self.y_mean[:, ph] = np.mean(y, axis=0)
            self.y_std[:, ph] = np.std(y, axis=0)

    def _fit_posterior_stats(self, u_trans_embeds, y_vecs, phases):
        y_preds = self.predict(u_trans_embeds, phases)
        yu_ground = get_embeddings_from_mem(y_vecs, phases)
        yu_expected = get_embeddings_from_mem(y_preds, phases)
        self.yu_variance = np.asarray(np.var(yu_ground - yu_expected, axis=0))
        self.d_yu_variance = np.asarray(
            np.var(
                (yu_ground[self.phase_n:] - yu_ground[:-self.phase_n]) - (
                            yu_expected[self.phase_n:] - yu_expected[:-self.phase_n])
                , axis=0))

    @staticmethod
    def _norm_derivative(y_std, u_std):
        """
        F(u, phi) = Sum[|phi=ph|] denorm^y_ph(f_ph(norm^u(u)))
        dF/du|phi=ph = ddenorm^y_ph/df_ph(norm^u(u)) * df_ph/d(norm^u(u)) *  dnorm^u(u)/du ... chain rule
        dF/du|phi=ph =  std^y_ph * df_ph/d(norm^u(u)) * 1/std^u
        @param y_std:
        @param u_std:
        @return: [std^y_ph/std^u| for all ph]; (y_dim, u_dim, motor_phase_n, sensory_phase_n)
        """
        y_dim, phase_n = y_std.shape
        u_dim, _ = u_std.shape
        derivative_norm = np.ones((y_dim, u_dim, phase_n, phase_n))
        _u_std = 1 / u_std
        _u_std[u_std < MultiPhaseModel.NUMERICAL_EPS] = 0  # if the std is too small then kill the derivative
        for sens_ph in range(phase_n):
            for motor_ph in range(phase_n):
                derivative_norm[:, :, motor_ph, sens_ph] = y_std[None, :, sens_ph].T * _u_std[:, motor_ph]
        return derivative_norm

    def fit(self, u_trans_embeds, y_vecs, phases):
        self.models = []
        # STATISTICS LEARNING
        self._fit_prior_stats(u_trans_embeds, y_vecs, phases)
        u_embeds_flat = self.u_prepare(u_trans_embeds)
        for ph in range(self.phase_n):
            ph_sel = (phases[:, ph] == 1)
            X = u_embeds_flat[ph_sel]
            y = self.norm_y(y_vecs[ph_sel], ph)
            ##
            if np.sum(ph_sel) < self.degenerate_threshold:
                self.models.append(self.degenerate_model_builder())
                print("Warning: Phase={} has insufficient datasize={}. Degenerate model will be used.".format(
                    ph, np.sum(ph_sel)))
            else:
                self.models.append(self.model_builder())
            self.models[ph].fit(X, y)
        self._fit_posterior_stats(u_trans_embeds, y_vecs, phases)
        self._derivative_norm = self._norm_derivative(self.y_std, self.u_std)
        return self

    def predict_chronological(self, u_trans_embeds, phases):
        pass

    def prevariate(self, u_trans_embeds, phases):
        """
        Prediction of sensory variations conditioned by motor
        @param u_trans_embeds: [(transgait_window_size, u_dim, granularity), ..]
        @param phases:
        @return:
        """
        preds = np.zeros((len(u_trans_embeds), self.y_dim))
        for ph in range(len(self.models)):
            ph_args = np.where(phases[:, ph] == 1)[0]
            if len(ph_args) > 0:
                preds[ph_args, :] = self.yu_variance[:, ph]  # TODO one day this will be not static
        return preds

    def predict(self, u_trans_embeds, phases):
        """
        Prediction of gait responses for given phases
        @param u_trans_embeds: [(transgait_window_size, u_dim, granularity), ..]
        @param phases:
        @return:
        """
        u_embeds_flat = self.u_prepare(u_trans_embeds)
        preds = np.zeros((len(u_trans_embeds), self.y_dim))
        for ph in range(len(self.models)):
            ph_args = np.where(phases[:, ph] == 1)[0]
            if len(ph_args) > 0:
                preds_ph = self.phase_predict(u_embeds_flat[ph_args], ph)
                preds[ph_args, :] = preds_ph
        return preds

    def prevariate_gait_response(self, u_embeds_opt):
        """
        Same as prevariate but for gaits.

        @param u_embeds_opt: flattened embeddings or full non-trans(!) embeddings
        @return: sensory embedding (len(u_embeds_opt), self.y_dim, self.phase_n,)
        """
        return np.asarray([self.yu_variance] * len(u_embeds_opt))  # TODO one day this will be not static

    def predict_gait_response(self, u_embeds_opt):
        """
        Prediction of whole gait response.
        i.e. expected sensory response during the gait-cycle driven by given gait
        The single gait is transformed into transgait setup by assuming that the gait is repeated all the time.

        @param u_embeds_opt: flattened embeddings or full non-trans(!) embeddings
        @return: sensory embedding (len(u_embeds_opt), self.y_dim, self.phase_n,)
        """
        # The single gait is transformed into multiple (same) gaits
        if u_embeds_opt.ndim == 3:  # embedded gait input
            flt = self.norm_flattened_u(flatten(u_embeds_opt))
            u_embeds_flat = np.concatenate([flt] * self.transgait_window)
        elif u_embeds_opt.ndim == 2:  # flattened gait input
            u_embeds_flat = [np.concatenate([self.norm_flattened_u(u_embeds_opt[i])] * self.transgait_window) for i in
                             range(len(u_embeds_opt))]
        else:
            raise ValueError("Wrong embedding shape {}".format(u_embeds_opt.shape))

        preds = np.zeros((len(u_embeds_flat), self.y_dim, self.phase_n,))
        for ph in range(len(self.models)):
            preds_ph = self.phase_predict(u_embeds_flat, ph)
            preds[:, :, ph] = preds_ph
        return preds

    def manifesto_dict(self):
        ret = {
            "y_dimension": self.y_dim,
            "u_dimension": self.u_dim,
            "phase_number": self.phase_n,
            "degenerate_datasize_threshold": self.degenerate_threshold,
            "models": [mod.__class__.__name__ for mod in self.models],
            "transgait_window": self.transgait_window,
            "version": 3
        }
        return ret

    def save(self, path, name, force=False, annotation="Usual model"):
        dict_path = os.path.join(path, name)
        if os.path.exists(dict_path):
            if force:
                print("WARNING: The existing model {} is forced to overwrite.".format(dict_path))
            else:
                raise FileExistsError("The model {} already exists!".format(dict_path))
        else:
            os.mkdir(dict_path)

        # saving models
        for ph, mod in enumerate(self.models):
            ph_path = os.path.join(dict_path, str(ph))
            if not os.path.exists(ph_path):
                os.mkdir(ph_path)
            mod.save(ph_path)

        # saving stats
        np.savetxt(os.path.join(dict_path, "u_mean"), self.u_mean)
        np.savetxt(os.path.join(dict_path, "u_std"), self.u_std)
        np.savetxt(os.path.join(dict_path, "y_mean"), self.y_mean)
        np.savetxt(os.path.join(dict_path, "y_std"), self.y_std)
        np.savetxt(os.path.join(dict_path, "yu_variance"), self.yu_variance)
        np.savetxt(os.path.join(dict_path, "d_yu_variance"), self.d_yu_variance)

        # saving manifesto
        manifesto = self.manifesto_dict()
        manifesto["annotation"] = annotation
        manifesto["timestamp"] = time.time()

        file = open(os.path.join(dict_path, "manifesto.yaml"), 'w')
        file.write(yaml.dump(manifesto))
        file.close()

    @classmethod
    def load(cls, path, name):
        dict_path = os.path.join(path, name)
        manifesto = ""
        with open(os.path.join(dict_path, "manifesto.yaml"), "r") as file:
            manifesto = yaml.load(file, Loader=yaml.FullLoader)
        print("Loading {} with manifesto: {}".format(dict_path, manifesto))

        ##

        def non_builder():
            raise NotImplemented("Loaded model is read-only!")

        model = cls(
            model_builder=non_builder,
            degenerate_model_builder=non_builder,
            degenerate_datasize_threshold=manifesto["degenerate_datasize_threshold"],
            y_dimension=manifesto["y_dimension"],
            u_dimension=manifesto["u_dimension"],
            phase_number=manifesto["phase_number"]
        )
        if "transgait_window" in manifesto:
            model.transgait_window = manifesto["transgait_window"]
        else:
            print("WARNING: transgait_window not in manifesto. Using default value 0.")
            model.transgait_window = 0
        ##
        model.u_mean = np.loadtxt(os.path.join(dict_path, "u_mean"))
        model.u_std = np.loadtxt(os.path.join(dict_path, "u_std"))
        model._u_mean_flat = flatten(model.u_mean)
        model._u_std_flat = flatten(model.u_std)
        if "version" in manifesto:
            model.y_mean = np.loadtxt(os.path.join(dict_path, "y_mean"))
            model.y_std = np.loadtxt(os.path.join(dict_path, "y_std"))
            model.yu_variance = np.loadtxt(os.path.join(dict_path, "yu_variance"))
            if manifesto["version"] == 3:
                model.d_yu_variance = np.loadtxt(os.path.join(dict_path, "d_yu_variance"))
            else:
                model.d_yu_variance = model.yu_variance * 2  # Approximation var<F(d_u)-d_y> ~ 2*var<F(u)-y>
        else:  # backward compatibility
            model.norm_u = lambda x: x
            model.denorm_u = lambda x: x
            model.norm_flattened_u = lambda x: x
        ##
        model._derivative_norm = model._norm_derivative(model.y_std, model.u_std)
        for ph, model_name in enumerate(manifesto["models"]):
            ph_path = os.path.join(dict_path, str(ph))
            model.models.append(MODEL_LOADERS[model_name](ph_path))
        return model


def get_mem_from_record(record, a_name="a", u_name="u", y_name="y"):
    """

    @param record:
    @param a_name:
    @param u_name:
    @param y_name:
    @return: u_mem, y_mem, seg
    """
    u = record[u_name]
    y = record[y_name]
    a = record[a_name]

    # create MEM embeddings
    u_mem, seg = mean_by_segments(u, a)
    y_mem, _ = mean_by_segments(y, a)
    return u_mem, y_mem, seg


def get_embedding(memory, phases, upper_bound_index):
    """
    Returns phase ordered period of the memory (phase embedding).
    :param np.ndarray memory: time sequence of measurement vectors
    :param np.ndarray phases: time sequence of phase indicators
    :param int upper_bound_index: time after the latest measurement vector present in embedding
    :return:
    """
    mem_part = memory[upper_bound_index - phases.shape[1]:upper_bound_index, :]
    phs_part = phases[upper_bound_index - phases.shape[1]:upper_bound_index, :]
    return mem_part.T.dot(phs_part)


def get_embeddings_from_mem(mem, phases):
    return np.asarray([get_embedding(mem, phases, i) for i in range(phases.shape[1], phases.shape[0] + 1)])


def get_data_to_fit(u_mem, y_mem, a, transgait_window_size=0, default_pad=None):
    """
    Preprocessing the u_mem and y_mem to a format for model fit and predict input.
    @param u_mem:
    @param y_mem:
    @param a:
    @return:
    """
    u_embs = get_embeddings_from_mem(u_mem, a)
    if default_pad is None:
        default_pad = np.zeros(u_embs[0].shape)
    u_trans_embs = get_transembeddings_from_embedding_chronology(u_embs, transgait_window_size)
    y_vecs = y_mem[len(a[0]) - 1:]
    phases = a[len(a[0]) - 1:]
    return u_trans_embs, y_vecs, phases


def sensory_continuous_prediction(forward_model: MultiPhaseModel, motor_signal, phase_signal):
    motor_embedder = E.Embedder(
        dimension=motor_signal.shape[1],
        granularity=phase_signal.shape[1],
        # combiner=M.cpg_rbf_embedding.affine_combiner(old_state_affinity=.5)
        combiner=E.mean_combiner()
    )
    preds = []
    debug = []
    for it in range(len(phase_signal)):
        motor_embedder.step(
            phase_activation=phase_signal[it, :],
            signal=motor_signal[it, :]
        )
        embed = motor_embedder.current_embedding()
        debug.append(np.zeros(embed.shape) + embed)
        pred = forward_model.predict(
            u_trans_embeds=np.asarray([embed]),
            phases=phase_signal[it:it + 1, :]
        )
        preds.append(pred[0])
    return np.asarray(preds), debug


def flatten(embeddings: np.ndarray):
    if embeddings.ndim == 2:
        return embeddings.reshape((embeddings.shape[0] * embeddings.shape[1],), order='C')
    elif embeddings.ndim == 3:
        return embeddings.reshape((embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2]), order='C')
    else:
        raise ValueError("Wrong embedding shape {}".format(embeddings.shape))


def de_flatten(flat_embeddings: np.ndarray, dimension, granularity):
    if flat_embeddings.ndim == 1:
        return flat_embeddings.reshape((dimension, granularity), order='C')
    elif flat_embeddings.ndim == 2:
        return flat_embeddings.reshape((flat_embeddings.shape[0], dimension, granularity), order='C')
    else:
        raise ValueError("Wrong FLAT embedding shape {}".format(flat_embeddings.shape))


def flatten_trans_embedding(embeddings: np.ndarray):
    if embeddings.ndim == 3:
        return embeddings.reshape((
            embeddings.shape[0] * embeddings.shape[1] * embeddings.shape[2],), order='C')
    if embeddings.ndim == 4:
        return embeddings.reshape(
            (
                embeddings.shape[0],
                embeddings.shape[1] * embeddings.shape[2] * embeddings.shape[3]), order='C')
    else:
        raise ValueError("Wrong embedding shape {}".format(embeddings.shape))


def de_flatten_trans_embedding(flat_embeddings: np.ndarray, trans_gait_window_size, dimension, granularity):
    if flat_embeddings.ndim == 1:
        return flat_embeddings.reshape((trans_gait_window_size, dimension, granularity), order='C')
    elif flat_embeddings.ndim == 2:
        return flat_embeddings.reshape((flat_embeddings.shape[0], trans_gait_window_size, dimension, granularity),
                                       order='C')
    else:
        raise ValueError("Wrong FLAT embedding shape {}".format(flat_embeddings.shape))


def get_transembeddings_from_embedding_chronology(embeddings, transgait_window_size):
    return np.asarray(
        [get_transgait_window(embeddings, i, transgait_window_size) for i in range(len(embeddings))])


def get_transgait_window(embeddings, emb_idx, transgait_window_size):
    """
    From chronological sequence of embeddings creates transgait_embedding:
     [embeddings[embd_idx - (transgait_widnow_size - 1) * granularity],
      embeddings[embd_idx - 1 * granularity],embeddings[embd_idx]]

    @param embeddings: chronological series of single-gait embeddings
    @param emb_idx:
    @param transgait_window_size:
    @return: tensor (transgait_window_size, dimension, granularity)
    """

    granularity = embeddings[0].shape[1]
    included_gaits = emb_idx // granularity + 1
    padd = transgait_window_size - included_gaits
    start_id = emb_idx - granularity * (min(included_gaits, transgait_window_size) - 1)
    if padd < 0:
        return embeddings[[start_id + i * granularity for i in range(transgait_window_size)], :, :]
    else:
        return np.concatenate([np.zeros((padd, embeddings[0].shape[0], granularity)),
                               embeddings[[start_id + i * granularity for i in range(included_gaits)], :, :]])


def mean_by_segments(signal, phase_segments):
    segment_start = 0
    current_segment = np.argmax(phase_segments[0, :])
    ret = []
    seg = []
    for i in range(signal.shape[0]):
        if np.argmax(phase_segments[i, :]) != current_segment:  # and (i > segment_start):
            ret.append(np.mean(signal[segment_start: i], axis=0))
            seg.append(phase_segments[segment_start, :])
            current_segment = np.argmax(phase_segments[i, :])
            segment_start = i
    return np.asarray(ret), np.asarray(seg)


def median_by_segments(signal, phase_segments):
    segment_start = 0
    current_segment = np.argmax(phase_segments[0, :])
    ret = []
    seg = []
    for i in range(signal.shape[0]):
        if np.argmax(phase_segments[i, :]) != current_segment:  # and (i > segment_start):
            ret.append(np.median(signal[segment_start: i], axis=0))
            seg.append(phase_segments[segment_start, :])
            current_segment = np.argmax(phase_segments[i, :])
            segment_start = i
    return np.asarray(ret), np.asarray(seg)


def segment_sizes(phase_segments):
    segment_start = 0
    current_segment = np.argmax(phase_segments[0, :])
    ret = []
    seg = []
    for i in range(phase_segments.shape[0]):
        if np.argmax(phase_segments[i, :]) != current_segment:
            seg.append(phase_segments[segment_start, :])
            ret.append(i - segment_start)
            current_segment = np.argmax(phase_segments[i, :])
            segment_start = i
    return np.asarray(ret), np.asarray(seg)


def embeddings_to_signal(embeddings: np.ndarray, phase_activations: np.ndarray):
    return np.einsum("ndg,ng->nd", embeddings, phase_activations)


if __name__ == '__main__':
    # one embedding
    test_embed = np.random.randn(18 * 30).reshape((18, 30))
    res = de_flatten(flatten(test_embed), 18, 30)
    assert np.all(test_embed == res)

    # multiple
    test_embed = np.random.randn(1000 * 18 * 30).reshape((1000, 18, 30))
    res = de_flatten(flatten(test_embed), 18, 30)
    assert np.all(test_embed == res)


    def slow_get_embedding(memory, phases, index):
        emb = np.zeros((memory.shape[1], phases.shape[1]))
        mem_part = memory[index - phases.shape[1]:index, :]
        phs_part = phases[index - phases.shape[1]:index, :]
        phs_part_num = np.argmax(phs_part, axis=1)
        for ph in range(phases.shape[1]):
            sel = np.where(phs_part_num == ph)[0]
            emb[:, ph] = mem_part[sel, :]
        return emb


    test_mem = np.random.randn(18 * 300).reshape((300, 18))
    test_ph = np.concatenate([np.identity(10) for i in range(30)])

    for i in range(10, 300):
        expected = slow_get_embedding(test_mem, test_ph, i)
        result = get_embedding(test_mem, test_ph, i)
        assert np.all(expected == result)

    ## trans
    test_embed = np.random.randn(1000 * 5 * 18 * 30).reshape((1000, 5, 18, 30))
    res = de_flatten_trans_embedding(flatten_trans_embedding(test_embed), 5, 18, 30)
    assert np.all(test_embed == res)

    res = flatten_trans_embedding(test_embed)
    exp = np.asarray([np.concatenate([flatten(test_embed[i][j]) for j in range(5)]) for i in range(len(test_embed))])
    assert np.all(exp == res)
