from data_handler import DataHandler


def main():
    print('entering main')
    data_handler = DataHandler()
    latest_bars = data_handler.get_latest_bars('AAPL')
    print(type(latest_bars['AAPL']))
    print('done')
    

main()
