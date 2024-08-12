import numpy as np
import h5py

"""
Saving and loading records.
A record is a dictionary string: list(numpy.array).
List of records is refered to as 'records'.
It is used to save/load states of the model during the run.
"""


def merge_records(record, append_record):
    """
    Concatenates append_record values to the corresponding variables in record.
    @param record:
    @param append_record:
    @return:
    """
    for k in record:
        record[k] = np.concatenate((record[k], append_record[k]))


def merge_records_horizontal(record, append_record, append_record_key_psfx=""):
    """
    Merges append_record keys into record.
    @param record:
    @param append_record:
    @param append_record_key_psfx:
    @return:
    """
    for k in append_record:
        new_key = k + append_record_key_psfx
        if new_key in record:
            print("WARNING: the key {} already exist in the original record.".format(new_key))
        record[new_key] = append_record[k]


def numpyfy_record(raw_record, dtype=np.float32, t_array=None):
    record_np = dict([(k, []) for k in raw_record[0]])
    for row in raw_record:
        for k in record_np:
            record_np[k].append([row[k]])

    for k in record_np:
        try:
            record_np[k] = np.concatenate(record_np[k])
        except Exception as e:
            print("Failed to numpyfy key {}, skipping. Reason: {}".format(k, e))

    if t_array is not None:
        record_np["t"] = np.asarray(t_array)

    return record_np


def numpyfy_record_safe(raw_record):
    defaults = {}
    variable_numpy_length = {}
    for r in raw_record:
        for k in r:
            if k not in variable_numpy_length:
                variable_numpy_length[k] = False

            if type(r[k]) is np.ndarray:
                if k not in defaults:
                    defaults[k] = np.zeros(r[k].shape).astype(r[k].dtype)
                elif len(defaults[k]) != len(r[k]):
                    variable_numpy_length[k] = True
                    if len(defaults[k]) < len(r[k]):
                        defaults[k] = np.zeros(r[k].shape).astype(r[k].dtype)
            elif type(r[k]) is float:
                defaults[k] = 0.
            elif type(r[k]) is int:
                defaults[k] = 0
            elif type(r[k]) is bool:
                defaults[k] = False
            elif isinstance(r[k], np.generic):
                defaults[k] = type(r[k])(0)
            else:
                print(f"Warning: Unsupported record value [{k}] type ({type(r[k])}) will be not included.")

    record_np = dict([(k, []) for k in defaults])

    for row in raw_record:
        for k in record_np:
            if k in row:
                if variable_numpy_length[k]:
                    val = np.zeros(defaults[k].shape).astype(defaults[k].dtype)
                    val[:len(row[k])] = row[k]
                    record_np[k].append([val])
                else:
                    record_np[k].append([row[k]])
            else:
                record_np[k].append([defaults[k]])

    for k in record_np:
        try:
            record_np[k] = np.concatenate(record_np[k])
        except Exception as e:
            print("Failed to numpyfy key {}, skipping. Reason: {}".format(k, e))

    return record_np


def crop_record(record_np, start, end):
    cropped_record_np = {}
    for k in record_np:
        cropped_record_np[k] = record_np[k][start:end]
    return cropped_record_np


def subsample_record(record_np, sample_num):
    idxs = [i for i in range(sample_num)]
    for k in record_np:
        record_np[k] = record_np[k][idxs]
    return record_np


def print_record_shapes(record):
    for k in record:
        print("{}:          {}{}".format(k, record[k].shape,record[k].dtype))


def save_records(path, records):
    if isinstance(records, dict):
        records = [records]
    assert isinstance(records, list)
    assert isinstance(records[0], dict)
    with h5py.File(path, "w") as f:
        cou = 0
        for record in records:
            record_group = f.create_group(str(cou))
            cou += 1
            for k in record:
                data = record[k]
                if not isinstance(data, np.ndarray):
                    data = np.asarray(data)
                # data.astype(dtype=np.float32)
                record_group.create_dataset(k, shape=data.shape, dtype=data.dtype, data=data)


def load_records(path, to_working_memory=True):
    ret = []
    f = h5py.File(path, 'r')
    for record_key in f.keys():
        group = f[record_key]
        record = {}
        for var_key in group.keys():
            if to_working_memory:
                record[var_key] = np.asarray(group[var_key])
            else:
                record[var_key] = group[var_key]
        ret.append(record)
    if to_working_memory:
        f.close()
    return ret


def load_init_state_from_record(param_record, init_state, variables_to_set, from_iteration):
    if variables_to_set is None:
        variables_to_set = init_state.keys()
    for k in variables_to_set:
        if k not in param_record.keys():
            print("WARNING: no param {} in loaded init record. Its default init will be used.".format(k))
        else:
            init_state[k] = param_record[k][from_iteration]


def get_record_slice(record, slice_iter, exclude_terms=()):
    """
    Inits dynamic variables from the record into the given dynamic system descriptor.
    :param dict record:
    :param int slice_iter:
    :param tuple exclude_terms:
    :return:
    """
    slice = {}
    for var_name in record:
        if var_name not in exclude_terms:
            slice[var_name] = record[var_name][slice_iter]
    return slice


def max_len(record):
    length = 0
    for k in record:
        if len(record[k]) > length:
            length = len(record[k])
    return length
