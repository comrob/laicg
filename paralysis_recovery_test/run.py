"""Terminal access point for the paralysis recovery test."""

import argparse
import logging
from paralysis_recovery_test.evaluation import create_report
from paralysis_recovery_test.experiments import paralysis_and_recovery_experiment, undisturbed_walking_experiment, \
    train_walking_model_experiment, Variant
import os
from paralysis_recovery_test.configuration import (EXTRACT_PATH, TRANSFER_PATH, WALK_TRANSFER_COLLECTION_NAME,
                                                   UNDISTURBED_COLLECTION_NAME)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
LOG = logging.getLogger(__name__)

LEARN_MODEL = "learn"
TUNE_THRESHOLD = "tune"
TEST_ADAPTIVITY = "test"
EVALUATE_BANNER = "eval"
DEFAULT_TEST_COLLECTION_NAME = "paralysis_recovery"


def assert_walk_model(banner_name):
    model_path = os.path.join(TRANSFER_PATH, banner_name, WALK_TRANSFER_COLLECTION_NAME)
    assert os.path.exists(
        model_path), f"Missing walking model at {model_path}. The model learning must be performed first."


def check_threshold_tune(banner_name):
    pth = os.path.join(EXTRACT_PATH, banner_name, UNDISTURBED_COLLECTION_NAME)
    if os.path.exists(pth):
        return banner_name
    LOG.warning(f"Missing tuning data at {pth}. The default zero-model threshold will be used.")
    return None


def process_parse(args):
    target_banner = args.banner_name
    transfer_banner = target_banner

    if args.transfer_banner is not None:
        transfer_banner = args.transfer_banner[0]

    if args.action_type == LEARN_MODEL:
        return train_walking_model_experiment(target_banner, context_steps=5, overwrite=True)
    elif args.action_type == TUNE_THRESHOLD:
        assert_walk_model(transfer_banner)
        return undisturbed_walking_experiment(target_banner, transfer_banner, context_steps=3, overwrite=True)
    elif args.action_type == TEST_ADAPTIVITY:
        assert_walk_model(transfer_banner)
        threshold_banner = check_threshold_tune(transfer_banner)
        return paralysis_and_recovery_experiment(
            target_banner, DEFAULT_TEST_COLLECTION_NAME, transfer_banner_name=transfer_banner,
            variant=Variant.REACTIVE, undisturbed_measurement_banner_name=threshold_banner, context_steps=5,
            overwrite=True)
    elif args.action_type == EVALUATE_BANNER:
        assert_walk_model(transfer_banner)
        return create_report(target_banner)


def main():
    parser = argparse.ArgumentParser(description='Paralysis-recovery experiment and evaluation pipeline.')
    parser.add_argument('banner_name',
                        help="Identifies a banner: a set of experimental collections which are evaluated together.")
    parser.add_argument(
        'action_type', choices=[LEARN_MODEL, TUNE_THRESHOLD, TEST_ADAPTIVITY, EVALUATE_BANNER],
        help=f"Type one of four following actions. "
             f"'{LEARN_MODEL}': Experimental collection learning a walking model (is necessary). "
             f"'{TUNE_THRESHOLD}': Experimental collection with undisturbed navigation for parameter tuning (optional). "
             f"'{TEST_ADAPTIVITY}': Experimental collection testing the paralysis and recovery adaptation. "
             f"'{EVALUATE_BANNER}': Creates report for all collections in the banner. "
    )
    parser.add_argument(
        '--transfer', nargs=1, action='store', dest="transfer_banner", required=False,
        help=f"'--transfer X' will use collections {LEARN_MODEL} and {TUNE_THRESHOLD} from banner named X."
    )

    ##
    args = parser.parse_args()
    process_parse(args)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.INFO)
    main()
