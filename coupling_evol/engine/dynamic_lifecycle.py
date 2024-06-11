import numpy as np

from coupling_evol.engine.embedded_control import EmbeddedTargetParameter
from coupling_evol.agent.components.internal_model import forward_model as FM
import os
import coupling_evol.engine.common as C
from typing import Callable, List

from coupling_evol.agent.components.internal_model.regressors import ModelInterface, JustMean


class WorldModel(object):
    """
    Represents ensemble of models.
    """
    MODEL = "_fwm"

    def __init__(self, directory_path: str, regressor_builder: Callable[[], ModelInterface],
                 transgait_window_size=1, force_overwrite=False
                 ):
        self.force_overwrite = force_overwrite
        if not force_overwrite:
            self._ensemble_size = self.ensemble_size_from(directory_path)
        else:
            self._ensemble_size = 0

        if self._ensemble_size > 0:
            self._models = [FM.MultiPhaseModel.load(directory_path, self.context_name(i + 1)) for i in
                            range(self._ensemble_size)]
        else:
            self._models = []
        self.regressor_builder = regressor_builder
        self.transgait_window_size = transgait_window_size
        self.directory_path = directory_path

    @classmethod
    def create_with_models(
            cls, models: List[FM.MultiPhaseModel],
            directory_path: str, regressor_builder: Callable[[], ModelInterface],
            transgait_window_size=1, force_overwrite=False):
        ret = cls(
            directory_path=directory_path, regressor_builder=regressor_builder,
            transgait_window_size=transgait_window_size, force_overwrite=force_overwrite)
        for m in models:
            ret.append(m)
        return ret

    def append(self, model: FM.MultiPhaseModel):
        self._models.append(model)
        self._ensemble_size += 1
        model.save(
            self.directory_path,
            self.context_name(self._ensemble_size),
            force=self.force_overwrite,
            annotation=f"WorldModel origin dir {self.directory_path} and ctx {self._ensemble_size}."
        )

    def learn_and_append(self, phase_signal, u_signal, y_signal):
        new_model = self.learn_model_with_regressor(
            phase_signal, u_signal, y_signal,
            regressor_builder=self.regressor_builder, transgait_window_size=self.transgait_window_size
        )
        self.append(new_model)

    @property
    def models(self):
        return self._models

    def __len__(self):
        return len(self._models)

    @classmethod
    def context_name(cls, ctx_id: int):
        return str(ctx_id) + cls.MODEL

    @classmethod
    def model_context_ids(cls, directory_path):
        ids = []
        for fn in os.listdir(directory_path):
            if cls.MODEL in fn:
                ids.append(int(fn.split('_')[0]))
        return sorted(set(ids))

    @classmethod
    def ensemble_size_from(cls, directory_path):
        return len(cls.model_context_ids(directory_path))

    @staticmethod
    def learn_model_with_regressor(
            phase_signal, u_signal, y_signal,
            regressor_builder: Callable[[], ModelInterface], transgait_window_size=1
    ) -> FM.MultiPhaseModel:

        record = {
            "a": phase_signal,
            "u": u_signal,
            "y": y_signal
        }
        # segmenting record into mem format
        u_mem, y_mem, seg = FM.get_mem_from_record(record,
                                                   a_name="a",
                                                   u_name="u",
                                                   y_name="y"
                                                   )

        # transforming mem format into list of embeddings
        u_embs, y_vecs, phases = FM.get_data_to_fit(u_mem, y_mem, seg,
                                                    transgait_window_size=transgait_window_size)

        # train the multi-phase model
        model = FM.MultiPhaseModel(
            model_builder=regressor_builder,
            degenerate_model_builder=lambda: JustMean(np.mean(y_vecs, axis=0)),
            y_dimension=y_mem.shape[1],
            u_dimension=u_mem.shape[1],
            phase_number=seg.shape[1],
            degenerate_datasize_threshold=10,
            transgait_window=transgait_window_size
        )
        model.fit(u_embs, y_vecs, phases)
        return model


class DynamicLifecycle(object):
    """
    Controls robot and learns the world model.
    """

    R_CMD = "u"
    R_CMD_T = "u_t"
    R_OBSERVATION = "y"
    R_TARGET = "y_trg"

    def __init__(self, world_model: WorldModel, sensor_dim: int, motor_dim: int, granularity: int):
        self.wm = world_model
        ##
        self.sensor_dim = sensor_dim
        self.motor_dim = motor_dim
        self.granularity = granularity

    def __call__(self, target: EmbeddedTargetParameter, observation: np.ndarray) -> C.COMMAND_T:
        pass
