from dataclasses import dataclass
from datetime import datetime
from marshmallow import Schema, fields

from ..utils import marshal_date


@dataclass
class AssetBase:

    symbol: str = None
    open: float = None
    high: float = None
    low: float = None
    volume: float = None

    @classmethod
    def create(cls, **kwargs):
        data = {}
        for k, v in kwargs.items():
            if hasattr(cls, k):
                data[k] = v
        return cls(**data)


@dataclass
class Asset(AssetBase):

    close: float = None
    adjusted_close: float = None
    dividend_amount: float = None
    split_coefficient: float = None
    date: datetime = None

    @property
    def price(self):
        return self.close


@dataclass
class QuoteAsset(AssetBase):

    price: float = None
    latest_trading_day: datetime = None


class BaseSchema(Schema):

    symbol = fields.Str()
    open = fields.Float()
    high = fields.Float()
    low = fields.Float()
    volume = fields.Float()

    def marshal(func):
        def _marshal(self, data, *args, **kwargs):
            res = {}
            for k, v in data.items():
                k = BaseSchema.clean(k)
                res[k] = v
            return func(self, res, *args, **kwargs)
        return _marshal

    @marshal
    def load(self, data):
        return super(BaseSchema, self).load(data)

    @staticmethod
    def clean(key):
        split = key.split(' ')
        if len(split) > 2:
            return '_'.join(split[1:])
        return split[-1]


class AssetSchema(BaseSchema):

    close = fields.Float()
    adjusted_close = fields.Float()
    dividend_amount = fields.Float()
    split_coefficient = fields.Float()
    date = fields.Date()


class QuoteAssetSchema(BaseSchema):

    price = fields.Float()
    latest_trading_day = fields.Date()
