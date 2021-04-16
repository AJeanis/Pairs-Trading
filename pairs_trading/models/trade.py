import uuid 

from .assets import Asset

EQUITY_TRADE_ENUM = ['buy', 'sell']

class EquityTrade(object):

    def __init__(self, asset: Asset, shares: float, trade_type: str):
        self.id = uuid.uuid4()
        self._asset = asset
        self._shares = shares
        self.trade_type = trade_type
        
    @property
    def asset(self):
        return self._asset

    @property
    def shares(self):
        return self._shares

    @property
    def trade_type(self):
        return self._trade_type

    @trade_type.setter
    def trade_type(self, trade_type: str):
        if not trade_type.lower() in EQUITY_TRADE_ENUM:
            raise ValueError('Invalid Trade Type!')
        self._trade_type = trade_type

    @property
    def entry_point(self):
        return self.asset.close

    def calculate_value(self, current_price: float):
        return current_price * self.shares

    def calculate_profit(self, current_price: float):
        diff = (current_price - self.entry_point) * self.shares
        return diff if self.trade_type == 'buy' else diff * -1


        




    