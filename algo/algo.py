from .models.account import Account
from .models.trade import TradeHolder
from .pipelines.algorithms import PairsTradingPipeline
from .pipelines.execution import LongExecutionPipeline, ShortExecutionPipeline
from .pipelines.TimeSeriesRetrieverPipeline
from .cofig import config, Config

ALGO_SELECTION = {
    'pairs': PairsTradingPipeline(config.interval_one, config, interval_two)
}

class AlgoTradingPipeline:

    def __init__(self, config: Config):
        self._account = Account(config.starting_cash)
        self._algorithm = ALGO_SELECTION[config.algorithm_type]
        self._longs = LongExecutionPipeline()
        self._shorts = ShortExecutionPipeline()
        self._retriver = TimeSeriesRetrieverPipeline(config.symbols, config.lower_range, config.upper_range)

    @property 
    def account(self):
        return self._account 

    @property 
    def algorithm(self):
        return self._algorithm

    @property 
    def longs(self):
        return self._longs

    @property 
    def shorts(self):
        return self._shorts 

    @property 
    def retriever(self):
        return self._retriver

    def initialize(self):
        self._initialize_algo()

    def _initialize_algo(self):
        data_one, data_two = self.retriever.get_daily()
        self.algorithm.initialize(data_one, data_two)



