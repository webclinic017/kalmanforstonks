from data_handler import DataHandler, Stock


def main():
    print('entering main')
    apple = Stock('AAPL')
    apple.get_latest_bars()
    t, o, h, l, c = apple.get_bar_data()
main()
