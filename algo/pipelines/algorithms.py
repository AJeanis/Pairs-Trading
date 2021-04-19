import numpy as np

from typing import List, Union

from ..models.assets import Asset
from ..models.indicators import PriceRatioSimpleMovingAverage, PriceRatio


class PairsTradingPipeline(object):

    def __init__(self, interval_one, interval_two):
        self._sma_one = PriceRatioSimpleMovingAverage(interval_one)
        self._sma_two = PriceRatioSimpleMovingAverage(interval_two)
        self._price_ratio = PriceRatio(interval_two)

    @property
    def sma_one(self):
        return self._sma_one

    @property
    def sma_two(self):
        return self._sma_two

    @property
    def price_ratio(self):
        return self._price_ratio

    @property
    def std(self):
        return self.price_ratio.std()

    @property
    def zscore(self):
        return (self.sma_one.latest_sma - self.sma_two.latest_sma)/self.std

    def initialize(self, data_one: List[Asset], data_two: List[Asset]):
        self._sma_one.initialize(data_one, data_two)
        self._sma_two.initialize(data_one, data_two)
        self._price_ratio.initialize(data_one, data_two)

    def append(self, asset_one: Asset, asset_two: Asset):
        self._sma_one.append(asset_one, asset_two)
        self._sma_two.append(asset_one, asset_two)
        self._price_ratio.append(asset_one, asset_two)

    def marshal_trade(func):
        def _marshal_trade(self):
            trade_eval = func(self)
            if trade_eval:
                asset_one, asset_two = self.sma_one.metadata[0]
                trade_type_one, trade_type_two = trade_eval
                return ({'symbol': asset_one.symbol, 'trade_type': trade_type_one},
                        {'symbol': asset_two.symbol, 'trade_type': trade_type_two})
            return None
        return _marshal_trade

    @marshal_trade
    def evaluate_trade(self):
        if self.zscore >= 1:
            print(self.zscore)
            return ('short', 'long')
        if self.zscore <= -1:
            print(self.zscore)
            return ('long', 'short')
