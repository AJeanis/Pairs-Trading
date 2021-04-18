from .trade import EquityTrade


class Account:

    def __init__(self, starting_cash):
        self._starting_cash = starting_cash
        self._cash = starting_cash

    @property
    def starting_cash(self):
        return self._starting_cash

    @property
    def cash(self):
        return self._cash

    def execute_trade(self, trade: EquityTrade):
        self._cash -= trade.entry_value

    def close_trade(self, trade: EquityTrade):
        self._cash += trade.closed_profit
