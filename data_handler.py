from abc import ABC, abstractmethod
import alpaca_trade_api as tradeapi



class DataHandler():
    def __init__(self):
        APCA_API_KEY_ID = "PKSYM3GWCJL9APB25QVI"
        APCA_API_SECRET_KEY = "ovLYSdPmNGfgUMoTfY8tmfgchjEywUlAIBtOWXWm"
        APCA_API_BASE_URL = "https://paper-api.alpaca.markets"
        APCA_API_DATA_URL = "https://data.alpaca.markets/"
        self.api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, 'v2')
        
        print('test')
    def get_latest_bars(self, symbol, N=1):
        barset = self.api.get_barset(symbol, 'day', limit=30)
        return barset
        
        
        
