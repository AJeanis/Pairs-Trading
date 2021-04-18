from dataclasses import dataclass
from datetime import datetime

class AssetMixin: 

    @classmethod 
    def create(cls, **kwargs):
        data = {}
        for k, v in kwargs.items():
            k = AssetMixin.clean(k)
            if hasattr(cls, k):
                data[k] = v 
        return cls(**data)

    @staticmethod 
    def clean(key): 
        split = key.split(' ')
        if len(split) > 2: 
            return '_'.join(split[1:])
        return split[-1]

@dataclass
class Asset(AssetMixin):

    symbol: str = None 
    open: float = None 
    close: float = None 
    volume: float = None 
    high: float = None 
    low: float = None
    close: float = None
    date: datetime = None

    @property 
    def price(self):
        return self.close

@dataclass 
class QuoteAsset(AssetMixin):

    symbol: str = None 
    open: float = None 
    high: float = None 
    low: float = None
    price: float = None 
    volume: float = None 
    latest_trading_day: datetime = None 





    