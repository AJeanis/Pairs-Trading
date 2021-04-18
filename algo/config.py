import os

from dotenv import load_dotenv

load_dotenv()

class Config:

    alpha_api_key = os.getenv('ALPHA_API_KEY')
    starting_cash = 100000
    interval_one = 20 
    interval_two = 50
    stop_loss = .5
    algorithm_type = 'pairs'
    symbols = ['GOOG', 'QQQ']
    lower_range = '04-01-2019'
    upper_reange = None 


config = Config()