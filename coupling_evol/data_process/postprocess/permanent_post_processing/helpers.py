from __future__ import annotations
from typing import TypeVar, Generic, List, Tuple, Callable
from coupling_evol.data_process.postprocess.permanent_post_processing import common as C
import itertools
from coupling_evol.data_process.postprocess.lifecycle_data import LifeCycleRawData
import os
import re

PORTFOLIO_T = TypeVar('PORTFOLIO_T', bound=C.DataPortfolio)


class Tag(object):
    def __init__(self, trial=0, scenario="", lifecycle="", environment=""):
        self.trial = str(trial)
        self.scenario = scenario
        self.lifecycle = lifecycle
        self.environment = environment

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([
                other.lifecycle == self.lifecycle,
                other.trial == self.trial,
                other.scenario == self.scenario,
                other.environment == self.environment,
            ])
        return False

    def __str__(self):
        return f"t{self.trial}_s{self.scenario}_lc{self.lifecycle}_en{self.environment}"


class PortfolioAggregate(Generic[PORTFOLIO_T]):
    def __init__(self, database: List[Tuple[PORTFOLIO_T, Tag]]):
        self.agg = database

    def get(self, tag: Tag) -> PORTFOLIO_T:
        for portfolio, _tag in self.agg:
            if tag == _tag:
                return portfolio
        raise ValueError(f"The tag {tag} not found.")

    def __getitem__(self, item: Tag):
        return self.get(item)

    def group_by(self, tag_name: str) -> List[Tuple[str, PortfolioAggregate[PORTFOLIO_T]]]:
        return [(key, PortfolioAggregate[PORTFOLIO_T](list(group))) for key, group in
                itertools.groupby(self.agg, lambda tup: getattr(tup[1], tag_name))]

    def tag_range(self, tag_name: str):
        labels = []
        for _, tag in self.agg:
            labels.append(getattr(tag, tag_name))
        return sorted(set(labels))

    def lifecycle_range(self):
        return self.tag_range('lifecycle')

    def scenario_range(self):
        return self.tag_range('scenario')

    def environment_range(self):
        return self.tag_range('environment')

    def trial_range(self):
        return self.tag_range('trial')

    def process(self, results_path, collection_name_getter: Callable[[Tag], str]):
        for portfolio, tag in self.agg:
            coll_name = collection_name_getter(tag)
            dlcdp = LifeCycleRawData(results_path, coll_name, 20)
            portfolio.process_and_save(dlcdp)
            del dlcdp

    def __str__(self):
        return ",\n".join([str(tag) for _, tag in self.agg])


PARSER = Callable[[str], str]


def extract_path_tags(data_path: str,
                      trial_parser: PARSER,
                      scenario_parser: PARSER,
                      lifecycle_parser: PARSER,
                      environment_parser: PARSER
                      ) -> List[Tuple[str, Tag]]:
    path_tags = []
    for file_name in os.listdir(data_path):
        if file_name[0] == '_':
            continue
        file_pth = os.path.join(data_path, file_name)
        if os.path.isdir(file_pth):
            tag = Tag(
                trial=int(trial_parser(file_name)),
                scenario=scenario_parser(file_name),
                lifecycle=lifecycle_parser(file_name),
                environment=environment_parser(file_name)
            )
            path_tags.append((file_pth, tag))
    return path_tags


def get_aggregate(data_path: str,
                  portfolio_class: PORTFOLIO_T,
                  trial_parser: PARSER,
                  scenario_parser: PARSER,
                  lifecycle_parser: PARSER,
                  environment_parser: PARSER,
                  path_tag_filter: Callable[[Tuple[str, Tag]], bool] = lambda x: True
                  ) -> PortfolioAggregate[PORTFOLIO_T]:
    path_tags = extract_path_tags(data_path, trial_parser, scenario_parser, lifecycle_parser, environment_parser)
    tags = set(map(lambda x: str(x[1]), path_tags))
    assert len(tags) == len(path_tags), "Data path should contain portfolios with different tags."
    return PortfolioAggregate[portfolio_class](
        list(map(lambda p_t: (portfolio_class(p_t[0]), p_t[1]),
                 # list(filter(lambda p_t: not (p_t[0]).split('/')[-1][0] == '_', path_tags))))
                 list(filter(path_tag_filter, path_tags))))
    )


PARAM_STRING = re.compile(r'-?([a-zA-Z]+)((-?\d+(\+\d+)?)(-(-?\d+(\+\d+)?))*)?')
PARAM_VALUES = re.compile(r'(^(-?\d+(\+\d+)?)|(--?\d+(\+\d+)?))')
PARAM_VALUE = re.compile(r'(-?\d+)(\+\d+)?')


def parameter_value_parser(parameter_value: str):
    values = PARAM_VALUES.findall(parameter_value)
    parsed_values = []
    for i, value in enumerate(values):
        if i == 0:
            _value = value[0]
        else: # consequent values have separating "-"
            _value = value[0][1:]
        parse = PARAM_VALUE.findall(_value)[0]
        if len(parse[1]) > 1: #has exponent part
            v = int(parse[0]) * pow(10, -int(parse[1][1:]))
        else:
            v = float(int(parse[0]))
        parsed_values.append(v)
    return parsed_values


def parameter_parser(parameter_string: str):
    """
    The parameter strings have following format:
    XV-XV-XV
    where X stands for parameter name made of characters and V is an optional parametrization which has following format:
    (-)N+N-(-)N+N where N is a digit. A+B means A*10^{-B} .
    @param parameter_string:
    @return:
    """
    params = PARAM_STRING.findall(parameter_string)
    ret = {}
    for param in params:
        ret[param[0]] = parameter_value_parser(param[1])
    return ret



if __name__ == '__main__':
    parameter_parser("Trngob-gxy-200-0-mv-8-4-LegPar5-99+2-1")
    # parameter_value_parser("5--99+2--1")
