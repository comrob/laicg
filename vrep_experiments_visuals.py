import logging

from coupling_evol.data_process.postprocess.lifecycle_data import LifeCycleRawData
from vrep_experiments_confs import *
from coupling_evol.data_process.postprocess.permanent_post_processing import helpers as PPP_H, data_portfolios
from visuals import image_merger

LOG = logging.getLogger(__name__)

VISUALS_DIR = os.path.join("results", "simulation_visuals")
EXTRACT_DIR = os.path.join("results", "simulation_extract")
if not os.path.exists(VISUALS_DIR):
    os.mkdir(VISUALS_DIR)



SENSOR_TICKS = ("head_vel", "roll_vel", "pitch_vel", "yaw_vel")
MOTOR_TICKS = [str(i) for i in range(18)]
DATA_PROCESSOR_CLASS = LifeCycleRawData


def get_portfolio_aggregate(trials, exp_name, output_path):
    database = []
    for trial in trials:
        for trg_cnf_name in TARGET_CONFIGS:
            for lc_cnf_name in LIFECYCLE_CONFIGS:
                for env_cnf_name in ENVIRONMENT_CONFIGS:
                    coll_name = get_collection_name(exp_name, lc_cnf_name, env_cnf_name, trg_cnf_name)
                    coll_name += f"_r{trial}"
                    _output_path = os.path.join(output_path, coll_name)
                    if not os.path.exists(_output_path):
                        os.mkdir(_output_path)
                    print(f"Processing {coll_name}")
                    tag = PPP_H.Tag(trial=trial, scenario=trg_cnf_name, lifecycle=lc_cnf_name, environment=env_cnf_name)
                    portfolio = data_portfolios.LongevEvaluation(_output_path)
                    database.append((portfolio, tag))
    return PPP_H.PortfolioAggregate[data_portfolios.DecimJournalEvaluation](database)


def collection_name_from(exp_name):
    def getter(tag: PPP_H.Tag):
        coll_name = get_collection_name(exp_name, tag.lifecycle, tag.environment, tag.scenario)
        coll_name += f"_r{tag.trial}"
        return coll_name

    return getter


def main():
    banner_name = "040724_vrep"
    # exp_name = "hyp3"
    exp_name = "test"
    # banner_name = "journal_data_h36"
    # exp_name = "hypo36"

    trials = [10]
    # mode = [50]  # ALL
    # mode = [62]  # ALL
    mode = [91]  # ALL

    raw_data_path = os.path.join(EXP_ROOT_PATH, banner_name)
    extract_data_path = os.path.join(EXTRACT_DIR, banner_name)
    output_visuals_path = os.path.join(VISUALS_DIR, banner_name)
    data_output_path = raw_data_path

    if not os.path.exists(output_visuals_path):
        os.mkdir(output_visuals_path)
    if not os.path.exists(extract_data_path):
        os.mkdir(extract_data_path)

    if 50 in mode:
        for trial in trials:
            for trg_cnf_name in TARGET_CONFIGS:
                for lc_cnf_name in LIFECYCLE_CONFIGS:
                    for env_cnf_name in ENVIRONMENT_CONFIGS:
                        coll_name = get_collection_name(exp_name, lc_cnf_name, env_cnf_name, trg_cnf_name)
                        coll_name += f"_r{trial}"
                        # _output_path = os.path.join(output_path, coll_name)
                        # if not os.path.exists(_output_path):
                        #     os.mkdir(_output_path)
                        _extract_data_path = os.path.join(extract_data_path, coll_name)
                        if not os.path.exists(_extract_data_path):
                            os.mkdir(_extract_data_path)
                        print(f"Processing {coll_name}")
                        dlcdp = LifeCycleRawData(raw_data_path, coll_name)
                        data_portfolios.LongevEvaluation(_extract_data_path).process_and_save(dlcdp)

    if 62 in mode:
        from visuals.dataportfolio_view.detail_analysis.portfolio import generate_portfolio as generate_portfolio_detail
        from visuals.dataportfolio_view.longev_conference_evaluation.portfolio import generate_portfolio_detail as generate_portfolio_longev
        portfolio_agg = get_portfolio_aggregate(trials, exp_name, data_output_path)
        for trial in trials:
            for trg_cnf_name in TARGET_CONFIGS:
                for lc_cnf_name in LIFECYCLE_CONFIGS:
                    for env_cnf_name in ENVIRONMENT_CONFIGS:
                        coll_name = get_collection_name(exp_name, lc_cnf_name, env_cnf_name, trg_cnf_name)
                        coll_name += f"_r{trial}"
                        _extract_data_path = os.path.join(extract_data_path, coll_name)
                        _output_visuals_path = os.path.join(output_visuals_path, coll_name)
                        if not os.path.exists(_output_visuals_path):
                            os.mkdir(_output_visuals_path)
                        pa = data_portfolios.LongevEvaluation(_extract_data_path)
                        generate_portfolio_detail(output_path=_output_visuals_path, pa=pa, name="")
                        generate_portfolio_longev(output_path=_output_visuals_path, pa=pa, name="")

    if 91 in mode:
        # decim video
        from visuals.dataportfolio_view.decim_journal_evaluation.video import generate_video
        # input_path = "results/simulation_extract/040724_vrep/test_lc231129-econ22+3-sclr5+3-msevalt10-20-zeps43+10-dlca_envVrep_trgTrngo-gxy-100--100-mv-8-4_r9"
        input_path = "results/simulation_extract/040724_vrep/test_lc231129-econ22+3-sclr5+3-msevalt10-20-zeps43+10-dlca_envVrep_trgTrngob-gxy-100--100-mv-8-4-LegPar36Swf-1+1-7+1_r10"
        # input_path = input_paths["PW"][0]
        pa = data_portfolios.LongevEvaluation(input_path)
        output_path = "results/decim2023journal_visuals/video_modelerror"
        generate_video(output_path, pa)

if __name__ == '__main__':
    main()
