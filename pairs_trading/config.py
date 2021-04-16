import os

from dotenv import load_dotenv

load_dotenv()

class Config:

    os.environ['PYTHONPATH'] = os.getcwd()
    alpha_api_key = os.getenv('ALPHA_API_KEY')


config = Config()

