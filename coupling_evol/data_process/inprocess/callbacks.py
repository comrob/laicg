from coupling_evol.data_process.inprocess.record_logger import RECORD_T
from coupling_evol.engine.common import RecordNamespace as RN
from coupling_evol.engine.experiment_executor import R_T, R_DURAVG
import coupling_evol.agent.components.controllers.fep_controllers as FEP_CTR
from coupling_evol.engine.embedded_control import EmbeddedTargetParameter
from coupling_evol.agent.components.controllers.fep_controllers import WaveFepFusionController as WFF_CTR
from coupling_evol.agent.lifecycle.embedded_staged_lifecycle import EmbeddedStagedLC as ES_LC
from coupling_evol.agent.lifecycle.embedded_staged_lifecycle import BabblePerformanceAlternation as BPA_LC
from coupling_evol.agent.components.ensemble_dynamics.estimator_competition import \
    TwoStageAggregatedScoreCompetition as TSAGS_CM
from typing import Dict, Union, Callable, Any
import numpy as np
from coupling_evol.environments.coppeliasim.target_providers.navigation import R_POS_XYZ, R_POS_RPY, R_GOAL_XY


def lc_ctr(key: str):
    return key + "_" + BPA_LC.CONTROL_PSFX


def none(key: str):
    return key


def pref_post_cmb(d: Dict[str, Union[np.ndarray, float]],
                  pref: Callable[[str], str],
                  post: Callable[[str], str]) -> Callable[[str, Any], Any]:
    def cmb(key: str, default=None):
        k = pref(post(key))
        if k not in d:
            return default
        else:
            return d[k]

    return cmb


def competing_fep_callback(d: RECORD_T):
    # TODO instead of this standalone, this could be injected into record logger.
    if len(d) == 0:
        return
    ctrl = pref_post_cmb(d, RN.LIFE_CYCLE, lc_ctr)
    lcy = pref_post_cmb(d, RN.LIFE_CYCLE, none)
    exe = pref_post_cmb(d, RN.EXECUTOR, none)
    trge = pref_post_cmb(d, RN.TARGET_PROVIDER, none)
    t = exe(R_T, 0.)
    dur = exe(R_DURAVG, 0.)
    amplitudes_scale = ctrl(WFF_CTR.AMPLITUDES_SCALE_NAME, 0.)
    last_delta_magnitude = ctrl(FEP_CTR.DELTA_MAGNITUDE, 0.)
    performance_error = ctrl(FEP_CTR.ERR_PERFORMANCE, 0.)
    target_error_variance = ctrl(FEP_CTR.TARGET_ERROR_VARIANCE, 0.)
    likelihood_error = ctrl(FEP_CTR.LIKELIHOOD_ERROR, 0.)
    likelihood_error_variance = ctrl(FEP_CTR.LIKELIHOOD_ERROR_VARIANCE, 0.)

    xyz = trge(R_POS_XYZ, np.zeros(3, ))
    rpy = trge(R_POS_RPY, np.zeros(3, ))
    goal = trge(R_GOAL_XY, np.zeros(2, ))

    target_parameter = EmbeddedTargetParameter.read_from_record(d, RN.LIFE_CYCLE(ES_LC.R_TARGET))
    sec_stage_score = lcy(TSAGS_CM.R_SECOND_STAGE_SCORE, 0.)
    sel_model = lcy(TSAGS_CM.R_MODEL_SELECTION, 0)
    aggr_score = np.sum(lcy(TSAGS_CM.R_CONFIDENCE_AGGREGATE, 0.) * target_parameter.weight) / np.sum(
        target_parameter.weight)
    print(f"T {t:5.1f}({dur * 1000:1.1f})| mgnt:{last_delta_magnitude:2.3f}"
          f" amp:{np.linalg.norm(WFF_CTR.amplitude(amplitudes_scale)):2.3f}"
          f" prf_mse:{performance_error:2.3f}({np.mean(target_error_variance):2.3f})"
          f" mod_mse:{np.mean(np.square(likelihood_error)):2.3f}({np.mean(likelihood_error_variance):2.3f})"
          f"| trg: v ({target_parameter.value[0, 0]:1.1f},{target_parameter.value[4, 0]:1.1f}) r {target_parameter.value[3, 0]:1.1f}"
          f"| pos: [{xyz[0]:1.1f},{xyz[1]:1.1f};{rpy[2]:1.1f}] g:[{goal[0]:1.1f},{goal[1]:1.1f}]"
          f"| mod:{sel_model:2d} sc2:{sec_stage_score:1.3f}({aggr_score:1.3f})"
          )
