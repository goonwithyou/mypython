import datetime as dt
import sys
import numpy as np


def dmy2weekday(dmy):
    return dt.datetime.strptime(str(dmy, encoding='utf-8'), '%d-%m-%Y')\
        .date().weekday()


def loadtxt(filename):
    weekday, close_prices = np.loadtxt(filename,
                                       delimiter=',',
                                       usecols=(1, 6),
                                       unpack=True,
                                       converters={1: dmy2weekday})
    return weekday, close_prices


def calc_average_prices(weekdays, close_prices):
    average_prices = np.zeros(5)
    for weekday in range(5):
        average_prices[weekday] = np.take(close_prices,
                                          np.where(weekdays == weekday)).mean()
    return average_prices


def main():
    weekdays, close_prices = loadtxt('./data/data/aapl.csv')
    # print(weekdays, close_prices)
    average_prices = calc_average_prices(weekdays, close_prices)
    # print(average_prices)
    max_index = np.argmax(average_prices)
    min_index = np.argmin(average_prices)
    s_week = ('MON', 'TUE', 'WEN', 'THU', 'FRI')
    for i, average_price in enumerate(average_prices):
        print(s_week[i], ':', average_price,
              '(max)' if i == max_index else
              '(min)' if i == min_index else '')
    return 0


if __name__ == '__main__':
    sys.exit(main())
