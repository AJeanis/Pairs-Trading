import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from typing import List, Union

from .assets import Asset, QuoteAsset

# TODO
# write a standard deviation class
# write a zscore class


class CalculationBase(object):

    def __init__(self, interval_size: int = 50):
        self.queue = []
        self.idx = 0
        self.metadata = []
        self.interval_size = interval_size

    def append(self, asset: Union[Asset, QuoteAsset]):
        raise NotImplementedError("Method not implemented!")

    def initialize(self, data: List[Union[Asset, QuoteAsset]]):
        if len(data) < self.interval_size:
            raise ValueError('Data not sufficient!')
        for asset in data:
            self.append(asset)

    def visualize(self):
        return NotImplementedError("Method not implemented")

    def top(self):
        return self.queue[-1]

    def first(self):
        return self.queue[self.idx]


class MultipleCalculationBase(CalculationBase):

    def append(self, asset_one: Union[Asset, QuoteAsset], asset_two: Union[Asset, QuoteAsset]):
        raise NotImplementedError("Method not implemented!")

    def initialize(self, data_one: List[Asset], data_two: List[Asset]):
        if len(data_one) != len(data_two):
            raise ValueError('Data lengths not matching!')
        if len(data_one) < self.interval_size:
            raise ValueError('Data not sufficient')
        for a, b in zip(data_one, data_two):
            self.append(a, b)


class SimpleMovingAverage(CalculationBase):

    def __init__(self, interval_size: int = 50):
        self.removed = 0
        super().__init__(interval_size)

    @property
    def latest_value(self):
        return self.top()[0]

    @property
    def latest_sma(self):
        return self.top()[1]

    @property
    def first_value(self):
        return self.first()[0]

    @property
    def first_sma(self):
        return self.first()[1]

    def sma(self):
        return [elem[1] for elem in self.queue]

    def append(self, asset: Union[Asset, QuoteAsset]):
        if (len(self.queue) - self.idx) == self.interval_size:
            self.removed += (self.first_value - self.removed)
            self.idx += 1
        price = asset.price
        curr_cum = price if not self.queue else price + self.latest_value
        self.queue.append([curr_cum, self.calculate(curr_cum)])
        self.metadata.append(asset)

    def calculate(self, curr_cum):
        return (curr_cum - self.removed)/((len(self.queue) - self.idx) + 1)


class PriceRatio(MultipleCalculationBase):

    def __init__(self, interval_size: int = 50):
        super().__init__(interval_size)

    def append(self, asset_one: Union[Asset, QuoteAsset], asset_two: Union[Asset, QuoteAsset]):
        if (len(self.queue) - self.idx) == self.interval_size:
            self.idx += 1
        price_ratio = asset_one.price/asset_two.price
        self.queue.append(price_ratio)
        self.metadata.append((asset_one, asset_two))

    def std(self):
        return np.std(self.queue[self.idx:])


class PriceRatioSimpleMovingAverage(SimpleMovingAverage, MultipleCalculationBase):

    def __init__(self, interval_size: int = 50):
        super().__init__(interval_size)

    def append(self, asset_one: Union[Asset, QuoteAsset], asset_two: Union[Asset, QuoteAsset]):
        if (len(self.queue) - self.idx) == self.interval_size:
            self.removed += (self.first_value - self.removed)
            self.idx += 1
        p_ratio = asset_one.price/asset_two.price
        curr_cum = p_ratio if not self.queue else p_ratio + self.latest_value
        self.queue.append([curr_cum, self.calculate(curr_cum)])
        self.metadata.append((asset_one, asset_two))

    def visualize(self):
        a_data, b_data = [a.price for a, b in self.metadata], [b.price for a, b in self.metadata]
        plt.plot(self.sma())
