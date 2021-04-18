from typing import List, Dict

from ..utils import marshal_date
from ..models.bases import TimeSeries 
from ..models.assets import Asset, QuoteAsset

class RetriverMixin: 

    def marshal_output(self, res: Dict, **kwargs):
        output = []
        min_date = marshal_date(min(res.keys()))
        max_date = marshal_date(max(res.keys())) 
        for k, v in res.items():
            date = marshal_date(k)
            if (self.lower_range or min_date) <= date <= (self.upper_range or max_date):
                v['date'] = date 
                v = {**v, **kwargs}
                asset = Asset.create(**v)
                output.append(asset)
        return sorted(output, key=lambda item: item.date)
            
class RetrieverPipeline(RetriverMixin): 

    def __init__(self, symbols: List[str], lower_range: str = None, upper_range: str = None):
        self.lower_range = None if not lower_range else marshal_date(lower_range) 
        self.upper_range = None if not upper_range else marshal_date(upper_range)
        self.symbols = [symbol.upper() for symbol in symbols]

class TimeSeriesRetrieverPipeline(RetrieverPipeline):

    def __init__(self, symbols: List[str], lower_range: str = None, upper_range: str = None):
        super().__init__(symbols, lower_range, upper_range)
        self.retriever = TimeSeries()

    def get_daily(self):
        res = {}
        for symbol in self.symbols: 
            output, _ = self.retriever.get_daily_adjusted(symbol, outputsize='full')
            output = self.marshal_output(output, symbol=symbol)
            res[symbol] = output
        return res

    def get_quote(self):
        res = {}
        for symbol in self.symbols: 
            output, _ = self.retriever.get_quote_endpoint(symbol)
            res[symbol] = QuoteAsset.create(**output)
        return res

    


            



    