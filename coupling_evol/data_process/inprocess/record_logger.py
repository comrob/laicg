import numpy as np
from coupling_evol.engine.common import RecordType
import coupling_evol.data_process.inprocess.records as R
import os
import logging as log
import coupling_evol.engine.common as C
from typing import List, Dict, Union, Callable
from multiprocessing import Process
from abc import ABC

VALUE_T = Union[float, np.ndarray, List[np.ndarray], List[float]]
KEY_T = str
RECORD_T = Dict[KEY_T, VALUE_T]
RECORDER_T = Callable[[KEY_T, VALUE_T], None]
CALLBACK_T = Callable[[RECORD_T], None]


class Paths(object):
    NP_RECORD = "_snap.hdf5"

    def __init__(self, directory_path):
        self.directory_path = directory_path

    def numpy_record_snap(self, snap_id: int):
        return str(snap_id) + self.NP_RECORD

    def path(self, file_name):
        return os.path.join(self.directory_path, file_name)

    def get_snap_ids(self):
        ids = []
        for fn in os.listdir(self.directory_path):
            if self.NP_RECORD in fn:
                ids.append(int(fn.split('_')[0]))
        return sorted(ids)

    def get_max_snap_id(self):
        ids = self.get_snap_ids()
        if len(ids) == 0:
            return 0
        return max(self.get_snap_ids())

    def validate(self):
        ids = self.get_snap_ids()
        return len(ids) == ids[-1]


def record_snap_save(path: str, numpy_buffer: List[Dict[str, Union[float, np.ndarray]]]):
    np_record = R.numpyfy_record_safe(numpy_buffer)
    R.save_records(path, np_record)


class Recordable(ABC):
    def __init__(self):
        self.callback: CALLBACK_T = lambda x: None
        self.callback_period = 1
        self.callback_counter = 0
        self.callback_is_set = False
        self.current_record = {}
        self.callback_permanent_record = {}

    def _callback_on_increment(self):
        if self.callback_is_set:
            self.callback_counter += 1
            if self.callback_counter >= self.callback_period:
                self.callback_counter = 0
                self.callback(self.callback_permanent_record)

    def record(self, key: str, val: VALUE_T):
        self.current_record[key] = val
        self.callback_permanent_record[key] = val

    @staticmethod
    def _copy_val(val: VALUE_T) -> VALUE_T:
        if type(val) is not np.ndarray:
            return np.asarray(val)
        return val.copy()

    def get_last_record(self) -> RECORD_T:
        pass

    def _increment(self):
        pass

    def increment(self):
        self._callback_on_increment()
        self._increment()
        self.current_record = {}

    def save_and_flush(self, wait_for_it=False):
        pass

    def set_callback(self, callback: CALLBACK_T, callback_period: int):
        self.callback_period = callback_period
        self.callback = callback
        self.callback_counter = 0
        self.callback_is_set = True


class NullRecorder(Recordable):

    def record(self, key: str, val: VALUE_T):
        pass

    def get_last_record(self) -> RECORD_T:
        return {}

    def _increment(self):
        pass

    def save_and_flush(self, wait_for_it=False):
        pass


class RecordLogger(Recordable):
    """
    Special class for recording and saving
    """

    def __init__(self, directory_path: str, force_override=False, buffer_max_size=100):
        super().__init__()
        self.buffer = []
        self.directory_path = directory_path
        self.pths = Paths(self.directory_path)
        self.buffer_max_size = buffer_max_size
        self._current_saver_process = None
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
            log.info(f"Creating directory {directory_path}")
        ##
        # create directory
        # get the snapshot count
        self.current_snapshot_count = self.pths.get_max_snap_id() + 1
        if self.current_snapshot_count > 1:
            log.info(f"Snaps in {directory_path} will be continued to snap id {self.current_snapshot_count}")

        self.force_override = force_override

    def get_last_record(self) -> Dict[str, Union[float, np.ndarray]]:
        if len(self.buffer) == 0:
            return {}
        return self.buffer[-1]

    def _increment(self):
        self.buffer.append(self.current_record)
        ###
        if self.buffer_size() >= self.buffer_max_size:
            self.save_and_flush()

    def _save_and_flush_numpy_dict(self):
        if self._current_saver_process is not None:
            self._current_saver_process.join()
        self._current_saver_process = Process(
            target=record_snap_save,
            args=(
                self.pths.path(self.pths.numpy_record_snap(self.current_snapshot_count)),
                self.buffer
            ))
        self._current_saver_process.start()
        self.buffer = []

    def buffer_size(self):
        return len(self.buffer)

    def save_and_flush(self, wait_for_it=False):
        if self.buffer_size() > 0:
            self._save_and_flush_numpy_dict()
            ##
            self.current_snapshot_count += 1
        if wait_for_it and self._current_saver_process is not None:
            self._current_saver_process.join()


class RecordAggregator(Recordable):

    def __init__(self):
        super().__init__()
        self.buffer = []
        ##

    def _increment(self):
        self.buffer.append(self.current_record)

    def get_last_record(self) -> Dict[str, Union[float, np.ndarray]]:
        if len(self.buffer) == 0:
            return {}
        return self.buffer[-1]

    def save_and_flush(self, wait_for_it=False):
        self.buffer = []


def postfix_adder(postfix: str):
    _postfix = ""
    if len(postfix) > 0:
        _postfix = "_" + postfix
    return lambda key: key + _postfix


def prefix_adder(postfix: str):
    _prefix = ""
    if len(postfix) > 0:
        _prefix = postfix + "_"
    return lambda key: _prefix + key


def select_prefix(record, prefix: str, postfix: str = ""):
    _prefix = ""
    _postfix = ""
    prf_check = lambda x: True
    psf_check = lambda x: True
    prf_cut = lambda x: x
    psf_cut = lambda x: x

    if len(prefix) > 0:
        _prefix = prefix + "_"
        prf_check = lambda x: _prefix.__eq__(x[:len(_prefix)])
        prf_cut = lambda x: x[len(_prefix):]

    if len(postfix) > 0:
        _postfix = "_" + postfix
        psf_check = lambda x: _postfix.__eq__(x[:len(_postfix)])
        psf_cut = lambda x: x[:-len(_postfix)]

    ret = {}
    for k in record:
        if prf_check(k) and psf_check(k):
            ret[prf_cut(psf_cut(k))] = record[k]
    return ret


def clear_snap_directory(directory_path):
    for fld in os.listdir(directory_path):
        if Paths.NP_RECORD in fld:
            os.remove(os.path.join(directory_path, fld))


def _extract_defaults(loaded_records: List[Dict[str, np.ndarray]]):
    defaults = {}
    lengths = []
    variable_row_shape = {}
    for rec in loaded_records:
        reclen = 0
        for k in rec:
            reclen = len(rec[k])
            if k not in defaults:
                defaults[k] = rec[k][0] * 0
                variable_row_shape[k] = False
            elif defaults[k].shape != rec[k][0].shape:
                variable_row_shape[k] = True

            if defaults[k].ndim >= 1 and defaults[k].shape[0] < rec[k][0].shape[0]:
                defaults[k] = rec[k][0] * 0
        lengths.append(reclen)
    return defaults, lengths, variable_row_shape


def _add_default_records_into_snaps(loaded_records: List[Dict[str, np.ndarray]]):
    defaults, lengths, variable_row_shape = _extract_defaults(loaded_records)
    ret = []
    for i, rec in enumerate(loaded_records):
        tmp = {}
        for k in defaults:
            if k not in rec:
                # print(f"WARNING!: The record key {k} is not present in {i}th loaded snap."
                #       f" Adding default elements.")
                # tmp[k] = np.asarray([defaults[k]] * lengths[i])
                data = np.asarray([defaults[k]] * lengths[i])
            else:
                data = rec[k]
                # tmp[k] = rec[k]

            if variable_row_shape[k]:
                curr_row_len = data[0].shape[0]
                tmp[k] = np.asarray([defaults[k]] * lengths[i])
                for rid in range(lengths[i]):
                    tmp[k][rid][:curr_row_len] = data[rid]
            else:
                tmp[k] = data
        ret.append(tmp)
    return ret


def np_record_snaps_load(directory_path):
    pth = Paths(directory_path)
    max_id = pth.get_max_snap_id()
    recs = [R.load_records(pth.path(pth.numpy_record_snap(i + 1)))[0] for i in range(max_id)]
    recs = _add_default_records_into_snaps(recs)
    ret = recs[0]
    for i in range(1, max_id):
        R.merge_records(ret, recs[i])
    return ret
