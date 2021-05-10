from abc import ABC, abstractmethod
import alpaca_trade_api as tradeapi
import numpy as np
import datetime as dt
from dotenv import load_dotenv
import os
import pandas as pd

class DataHandler(ABC):
    def __init__(self):
        load_dotenv()
        APCA_API_KEY_ID = os.environ.get('APCA_API_KEY_ID')
        APCA_API_SECRET_KEY = os.environ.get('APCA_API_SECRET_KEY')
        APCA_API_BASE_URL = os.environ.get('APCA_API_BASE_URL')
        APCA_API_DATA_URL = os.environ.get('APCA_API_DATA_URL')
        self.api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, 'v2')
        print(type(self.api))        
    def get_latest_bars(self, N=1):
        barset = self.api.get_barset(self.symbol, 'day', limit=30)
        return barset[self.symbol]
        

    
            
class Stock(DataHandler):
    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol
    
    def get_bar_data(self):
        barset = super().get_latest_bars()
        n = len(barset)
        timestamps = []
        opens = np.zeros(n)
        highs = np.zeros(n)
        lows = np.zeros(n)
        closes = np.zeros(n)
        i = 0
        for bar in barset:
            print(bar.t)
##            print(type(bar.t))
            timestamps.append( bar.t )
            opens[i] = bar.o
            highs[i] = bar.h
            lows[i] = bar.l
            closes[i] = bar.c
            i += 1
            
        return timestamps, opens, highs, lows, closes
