from typing import Union, Dict

from .models.assets import Asset, QuoteAsset
from .config import DevelopmentConfig
from .algo import AlgoTradingPipeline
from .pipelines.retriever import TimeSeriesRetrieverPipeline


class BacktestAlgoTradingPipeline(AlgoTradingPipeline):

    def __init__(self, development_config: DevelopmentConfig):
        super().__init__(development_config)
        self.init_data = None
        self.testing_data = None
        self._long_profit = None
        self._short_profit = None 
        self._long_value = None 
        self._short_value = None 

    @property 
    def long_profit(self):
        return self._long_profit

    @property 
    def long_value(self):
        return self._long_value 

    @property 
    def short_profit(self):
        return self._short_profit

    @property 
    def short_value(self):
        return self._short_value

    def initialize(self): 
        self._initialize_data()
        self._initialize_algo()

    def _initialize_data(self):
        max_interval = max(self.config.interval_one, self.config.interval_two)
        data = self.retriever.get_daily()
        data_one, data_two = [data[k] for k in data]
        self.init_data = [data_one[:max_interval], data_two[:max_interval]]
        self.testing_data = [data_one[max_interval:], data_two[max_interval:]]

    def _initialize_algo(self):
        data_one, data_two = self.init_data 
        self.algorithm.initialize(data_one, data_two)

    def trade(self, asset_dic: Dict[str, Union[Asset, QuoteAsset]]):
        self.evaluate(asset_dic)
        self.manage_risk(asset_dic)

    def assemble_asset_dic(self, data_one, data_two):
        return {self.config.symbols[0]: data_one, self.config.symbols[1]: data_two}

    def backtest(self):
        for data_one, data_two in zip(self.testing_data[0], self.testing_data[1]):
            asset_dic = self.assemble_asset_dic(data_one, data_two)
            self.trade(asset_dic)

    def results(self):
        self.evaluate_results()
        print('Long Profit:', self.long_profit)
        print('Long Value:', self.long_value)
        print('Short Profit', self.short_profit)
        print('Short Value', self.short_value)
        print('ROI:', (self.long_value + self.short_value)/self.account.starting_cash)

    def evaluate_results(self):
        data_one, data_two = self.testing_data[0][-1], self.testing_data[1][-1]
        asset_dic = self.assemble_asset_dic(data_one, data_two)
        self.evaluate_long(asset_dic)
        self.evaluate_short(asset_dic)

    def evaluate_long(self, asset_dic):
        self._long_profit = self.longs.calculate_profit(asset_dic)
        self._long_value = self.longs.calculate_value(asset_dic) 
        
    def evaluate_short(self, asset_dic):
        self._short_profit = self.shorts.calculate_profit(asset_dic)
        self._short_value = self.shorts.calculate_value(asset_dic)

