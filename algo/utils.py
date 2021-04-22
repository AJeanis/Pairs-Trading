from datetime import date

EQUITY_TRADE_ENUM = ['long', 'short']

def marshal_trade_type(trade_type: str()):
    if trade_type.lower() not in EQUITY_TRADE_ENUM:
        raise ValueError('Invalid trade type!')
    return trade_type

def marshal_date(str_date: str):
    year, month, day = [int(elem) for elem in str_date.split('-')]
    return date(year, month, day)



