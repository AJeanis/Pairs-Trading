import numpy as np

from typing import List 

from ..models.assets import Asset
from ..models.indicators import PriceRatioSimpleMovingAverage, PriceRatio

class TechnicalIndicatorsPipeline(object):

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
        return np.std(self.price_ratio.queue)

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
