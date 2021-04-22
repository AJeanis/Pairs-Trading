import uuid

from typing import Union, Dict

from ..models.trade import TradeHolder, EquityTrade
from ..models.assets import Asset, QuoteAsset


class TradeExecutionPipeline:

    def __init__(self):
        self.trade_holder = TradeHolder()
        self.closed_holder = TradeHolder()

    @staticmethod 
    def determine_shares(func):
        def _determine_shares(self, *args, **kwargs):
            trade_prop = kwargs.get('trade_prop') 
            cash = kwargs.get('cash')
            asset = [elem for elem in args if isinstance(elem, Asset) or isinstance(elem, QuoteAsset)][0]
            shares = (trade_prop * cash)/asset.price
            return func(self, asset=asset, shares=shares)
        return _determine_shares

    def execute_trade(self):
        raise NotImplementedError('Method not implemented!')

    def close_trade(self):
        raise NotImplementedError('Method not implemented!')

    def manage_risk(self, asset_dic: Union[Asset, QuoteAsset], stop_loss: float):
        trades = []
        for k, v in self.trade_holder.trades.items():
            asset = asset_dic[v.symbol]
            if v.evaluate_risk(asset.price, stop_loss):
                trades.append(self.close_trade(k, asset_dic))
        return trades

    def calculate_profit(self, asset_dic: Dict[str, Union[Asset, QuoteAsset]]):
        return self.trade_holder.calculate_profit(asset_dic) + self.closed_holder.calculate_profit(asset_dic)

    def calculate_value(self, asset_dic: Dict[str, Union[Asset, QuoteAsset]]):
        return self.trade_holder.calculate_value(asset_dic) + self.closed_holder.calculate_value(asset_dic)


class LongExecutionPipeline(TradeExecutionPipeline):

    @TradeExecutionPipeline.determine_shares
    def execute_trade(self, asset: Union[Asset, QuoteAsset], shares: float = 0):
        trade = EquityTrade(
            asset=asset,
            shares=shares,
            trade_type='long'
        )
        self.trade_holder.add_trade(trade)
        return trade

    def close_trade(self, trade_id: uuid.uuid4, current_price: float):
        trade = self.trade_holder.get_trade(trade_id)
        trade.close_trade(current_price)
        self.closed_holder.add_trade(trade)
        self.trade_holder.delete_trade(trade_id)
        return trade


class ShortExecutionPipeline(TradeExecutionPipeline):

    @TradeExecutionPipeline.determine_shares
    def execute_trade(self, asset: Union[Asset, QuoteAsset], shares: float = 0):
        trade = EquityTrade(
            asset=asset,
            shares=shares,
            trade_type='short'
        )
        self.trade_holder.add_trade(trade)
        return trade

    def close_trade(self, trade_id: uuid.uuid4, current_price: float):
        trade = self.trade_holder.get_trade(trade_id)
        trade.close_trade(trade_id)
        self.closed_holder.add_trade(trade)
        self.trade_holder.delete_trade(trade_id)
        return trade
