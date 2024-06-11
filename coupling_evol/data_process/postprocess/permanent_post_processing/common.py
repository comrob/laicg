from coupling_evol.data_process.postprocess.lifecycle_data import LifeCycleRawData
import pickle
import os
from typing import TypeVar, Generic, List
from coupling_evol.agent.components.internal_model.forward_model import MultiPhaseModel

"""

"""


class ProcessedData(object):
    def process(self, dlcdp: LifeCycleRawData):
        pass


DATA_T = TypeVar('DATA_T', bound=ProcessedData)


class ProcessedDataWrap(Generic[DATA_T]):
    def __init__(self, data_path, data: DATA_T):
        self.path = data_path
        self._data: DATA_T = data
        self._empty = True

    @property
    def data(self) -> DATA_T:
        if self._empty:
            self.load()
        return self._data

    def load(self):
        if not os.path.exists(self.path):
            f"there is nothing to load in {self.path}"
        else:
            file = open(self.path, 'rb')
            self._data = pickle.load(file)
            self._empty = False

    def save(self):
        assert not self._empty, f"cannot save empty data to {self.path}"
        file = open(self.path, 'wb')
        pickle.dump(self.data, file)
        del self._data
        self._data = None
        self._empty = True

    def process(self, dlcdp: LifeCycleRawData):
        self._data.process(dlcdp)
        self._empty = False


class ModelsWrap(ProcessedDataWrap):
    def __init__(self, data_path):
        super().__init__(data_path, None)
        self.path = data_path
        self._data: List[MultiPhaseModel] = []
        self._empty = True

    @property
    def data(self) -> List[MultiPhaseModel]:
        if self._empty:
            self.load()
        return self._data

    def load(self):
        assert os.path.exists(self.path), f"there is nothing to load in {self.path}"
        self._data = []
        for i in range(len(os.listdir(self.path))):
            self._data.append(MultiPhaseModel.load(self.path, name=f"{i}_fwd"))
        self._empty = False

    def save(self):
        assert not self._empty, f"cannot save empty data to {self.path}"
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        for i, model in enumerate(self.data):
            model.save(path=self.path, name=f"{i}_fwd", force=True, annotation="Extracted model")

        del self._data
        self._data = None
        self._empty = True

    def process(self, dlcdp: LifeCycleRawData):
        self._data = dlcdp.get_models()
        self._empty = False


class DataPortfolio(object):
    def __init__(self, data_path):
        self.data_path = data_path
        if not os.path.exists(data_path):
            os.mkdir(data_path)

    def process_and_save(self, dlcdp: LifeCycleRawData):
        attrs = vars(self)
        for k in attrs:
            if isinstance(attrs[k], ProcessedDataWrap):
                attrs[k].process(dlcdp)
                attrs[k].save()
