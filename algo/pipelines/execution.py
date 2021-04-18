from typing import Union

from ..models.trade import TradeHolder, EquityTrade
from ..models.assets import Asset, QuoteAsset


class TradeExecutionPipeline:

    def __init__(self):
        self.trade_holder = TradeHolder()
        self.closed_holder = TradeHolder()

    def execute_trade(self):
        raise NotImplementedError('Method not implemented!')

    def close_trade(self):
        raise NotImplementedError('Method not implemented!')


class LongExecutionPipeline(TradeExecutionPipeline):

    def execute_trade(self, asset: Union[Asset, QuoteAsset], shares: float):
        trade = EquityTrade(
            asset=asset,
            shares=shares,
            trade_type='long'
        )
        self.trade_holder.add_trade(trade)
        return trade

    def close_trade(self, trade_id, current_price):
        trade = self.trade_holder.get_trade(trade_id)
        trade.close_trade(current_price)
        self.closed_holder.add_trade(trade)
        self.trade_holder.delete_trade(trade_id)
        return trade


class ShortExecutionPipeline(TradeExecutionPipeline):

    def execute_trade(self, asset: Union[Asset, QuoteAsset], shares: float):
        trade = EquityTrade(
            asset=asset,
            shares=shares,
            trade_type='short'
        )
        self.trade_holder.add_trade(trade)
        return trade

    def close_trade(self, trade_id, current_price):
        trade = self.trade_holder.get_trade(trade_id)
        trade.close_trade(trade_id)
        self.closed_holder.add_trade(trade)
        self.trade_holder.delete_trade(trade_id)
        return trade
