from dataclasses import dataclass
from datetime import datetime

@dataclass
class Asset:

    ticker: str = None 
    open: float = None 
    close: float = None 
    volume: float = None 
    high: float = None 
    low: float = None
    close: float = None
    date: datetime = None

    @classmethod
    def create(cls, **kwargs):
        data = {}
        for k, v in kwargs.items():
            k = k.split('.')[-1].lstrip()
            if hasattr(cls, k):
                data[k] = v 
        return cls(**data)