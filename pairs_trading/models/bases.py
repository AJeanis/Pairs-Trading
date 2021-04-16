import requests 
import json 

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.techindicators import TechIndicators


from ..config import config

class TimeSeries(TimeSeries):

    def __init__(self, *args, **kwargs):
        super().__init__(key=config.alpha_api_key, *args, **kwargs)

class CryptoCurrencies(CryptoCurrencies):

    def __init__(self, *args, **kwargs):
        super().__init__(key=config.alpha_api_key, *args, **kwargs)

class ForeignExchange(ForeignExchange):

    def __init__(self, *args, **kwargs):
        super().__init__(key=config.alpha_api_key, *args, **kwargs)

class FundamentalData(FundamentalData):

    def __init__(self, *args, **kwargs):
        super().__init__(key=config.alpha_api_key, *args, **kwargs)

class SectorPerformances(SectorPerformances): 

    def __init__(self, *args, **kwargs):
        super().__init__(key=config.alpha_api_key, *args, **kwargs)

class TechIndicators(TechIndicators):

    def __init__(self, *args, **kwargs):
        super().__init__(key=config.alpha_api_key, *args, **kwargs)