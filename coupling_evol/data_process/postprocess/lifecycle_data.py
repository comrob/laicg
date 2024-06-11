from coupling_evol.engine.dynamic_lifecycle import WorldModel
import coupling_evol.data_process.inprocess.record_logger as RL
import os
import coupling_evol.agent.components.internal_model.forward_model as FM
import shutil
import logging
LOG = logging.getLogger(__name__)
class ExperimentExecutorPaths(object):
    SNAPS_NAME = "snaps"
    ENSEMBLE_NAME = "ensemble"
    CONTROLLER_NAME = "controller"
    MISC_NAME = "misc"  # miscellaneous - unimportant files of any kind

    def __init__(self, results_path, collection_name):
        self.results_path = results_path
        self.collection_name = collection_name
        self.root_path = os.path.join(results_path, collection_name)

    def snaps_path(self):
        return os.path.join(self.root_path, self.SNAPS_NAME)

    def ensemble_path(self):
        return os.path.join(self.root_path, self.ENSEMBLE_NAME)

    def controller_path(self):
        return os.path.join(self.root_path, self.CONTROLLER_NAME)

    def misc_path(self):
        return os.path.join(self.root_path, self.MISC_NAME)

    def mkdirs(self):
        for pth in [
            self.snaps_path(),
            self.ensemble_path(),
            self.controller_path(),
            self.misc_path()
        ]:
            if not os.path.exists(pth):
                os.mkdir(pth)

    def any_exists(self):
        """
        Returns true if some experiment directory has file/directory
        @return:
        @rtype:
        """
        for pth in [
            self.snaps_path(),
            self.ensemble_path(),
            self.controller_path(),
            self.misc_path()
        ]:
            if os.path.exists(pth) and os.listdir(pth):
                return True
        return False

    def purge_dirs(self):
        for pth in [
            self.snaps_path(),
            self.ensemble_path(),
            self.controller_path(),
            self.misc_path()
        ]:
            if os.path.exists(pth):
                shutil.rmtree(pth)
            os.mkdir(pth)


class LifeCycleRawData(object):

    def __init__(self, results_path, collection_name):
        self._paths = ExperimentExecutorPaths(results_path, collection_name)
        self._stage_record = RL.np_record_snaps_load(self._paths.snaps_path())
        ens_size = WorldModel.ensemble_size_from(self._paths.ensemble_path())
        self._models = [
            FM.MultiPhaseModel.load(
                self._paths.ensemble_path(), WorldModel.context_name(i + 1)) for i in range(ens_size)]

        if ens_size == 0:
            LOG.warning("There is no trained model.")




    def get_raw_record(self):
        return self._stage_record

    def get_models(self):
        return self._models
