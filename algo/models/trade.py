import uuid
from typing import Union

from .assets import Asset, QuoteAsset
from ..utils import marshal_trade_type


class EquityTrade(object):

    def __init__(self, asset: Union[Asset, QuoteAsset],
                 shares: float,
                 trade_type: str):
        self._id = uuid.uuid4()
        self._asset = asset
        self._shares = shares
        self._trade_type = marshal_trade_type(trade_type)
        self._closed = False
        self._closed_profit = None
        self._closed_value = None

    @property
    def id(self):
        return self._id

    @property
    def asset(self):
        return self._asset

    @property
    def shares(self):
        return self._shares

    @property
    def trade_type(self):
        return self._trade_type

    @property
    def closed(self):
        return self._closed

    @property
    def closed_profit(self):
        if self._closed_profit == None:
            raise ValueError('Please close trade first!')
        return self._closed_profit

    @property
    def closed_value(self):
        if self._closed_value == None:
            raise ValueError('Please close trade first!')
        return self._closed_value

    @property
    def entry_point(self):
        return self.asset.price

    @property 
    def entry_value(self):
        return self.entry_point * self.shares

    def close_trade(self, current_price):
        self._closed = True
        self._closed_profit = self.calculate_profit(current_price)
        self._closed_value = self.calculate_value(current_price)

    def calculate_value(self, current_price: float):
        return current_price * self.shares

    def calculate_profit(self, current_price: float):
        diff = (current_price - self.entry_point) * self.shares
        return diff if self.trade_type == 'buy' else diff * -1

# TODO
# Connect TradeHolder with database

class TradeHolder:

    def __init__(self):
        self.trades = {}

    @property
    def calculate_value(self, current_price):
        total = 0
        for v in self.trades.values():
            total += v.calculate_value()
        return total

    @property
    def calculate_profit(self, current_price):
        total = 0
        for v in self.trades.values():
            total += v.calculate_profit()
        return total

    def sort_by_profit(self, current_price, desc=True):
        return sorted(self.trades.items(),
                      key=lambda item: item[1].calculate_profit(current_price),
                      reverse=(desc == True))

    def sort_by_value(self, current_price, desc=True):
        return sorted(self.trades.items(),
                      key=lambda item: item[1].calculate_value(current_price),
                      reverse=(desc == True))

    def add_trade(self, trade: EquityTrade):
        if trade.id in self.trades: 
            raise ValueError('ID already exists!')
        self.trades[trade.id] = trade

    def get_trade(self, trade_id: uuid.uuid4):
        if trade_id in self.trades:
            return self.trades[trade_id]
        raise ValueError('Invalid Trade ID!')

    def delete_trade(self, trade_id: uuid.uuid4):
        if trade_id in self.trades:
            del self.trades[trade_id]
        raise ValueError('Invalid Trade ID!')
