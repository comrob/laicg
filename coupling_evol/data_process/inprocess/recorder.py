from coupling_evol.data_process.inprocess.record_logger import RECORD_T, RECORDER_T, KEY_T, VALUE_T, CALLBACK_T
from coupling_evol.data_process.inprocess.record_logger import Recordable, NullRecorder, RecordLogger

###
_NULL_RECORDER = NullRecorder()
_CURRENT_LOGGER: Recordable = _NULL_RECORDER


def reset(directory_path: str, force_override=False, buffer_max_size=100):
    global _CURRENT_LOGGER
    _CURRENT_LOGGER.save_and_flush(wait_for_it=True)
    _CURRENT_LOGGER = RecordLogger(directory_path=directory_path, force_override=force_override,
                                   buffer_max_size=buffer_max_size)


def record(key: KEY_T, value: VALUE_T):
    global _CURRENT_LOGGER
    _CURRENT_LOGGER.record(key, value)


def increment():
    global _CURRENT_LOGGER
    _CURRENT_LOGGER.increment()


def save_and_flush(wait_for_it=False):
    global _CURRENT_LOGGER
    _CURRENT_LOGGER.save_and_flush(wait_for_it=wait_for_it)


def get_last_record():
    global _CURRENT_LOGGER
    return _CURRENT_LOGGER.get_last_record()


def get_recorder(prefix: str = "", postfix: str = "") -> RECORDER_T:
    _prefix = ""
    _postfix = ""
    if len(prefix) > 0:
        _prefix = prefix + "_"
    if len(postfix) > 0:
        _postfix = "_" + postfix

    def _record(key: str, value: VALUE_T):
        record(_prefix + key + _postfix, value)

    return _record


def get_null_recorder() -> RECORDER_T:
    global _NULL_RECORDER
    return _NULL_RECORDER.record


def set_callback(callback: CALLBACK_T, callback_period: int):
    global _CURRENT_LOGGER
    _CURRENT_LOGGER.set_callback(callback, callback_period=callback_period)
