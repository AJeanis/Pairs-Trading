from datetime import datetime

EQUITY_TRADE_ENUM = ['long', 'short']

def marshal_trade_type(trade_type: str()):
    if trade_type.lower() not in EQUITY_TRADE_ENUM:
        raise ValueError('Invalid trade type!')
    return trade_type

def marshal_date(date: str):
    year, month, day = [int(elem) for elem in date.split('-')]
    return datetime(year, month, day)
