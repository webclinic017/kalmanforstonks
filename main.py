from data_handler import DataHandler, Stock
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import math

def main():
    print('entering main')
    apple = Stock('AAPL')
    apple.get_latest_bars()
    t, o, h, l, c = apple.get_bar_data()
    ts = pd.Series(o, index = t)
    fig, ax = plt.subplots()
    ts.plot();
    plt.show()
    print("we did it")
main()
