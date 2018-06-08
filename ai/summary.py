import datetime as dt
import sys
import csv
import numpy as np


def dmy2weekday(dmy):
    return dt.datetime.strptime(str(dmy, encoding='utf-8'), '%d-%m-%Y')\
        .date().weekday()


def loadtxt(filename):
    weekdays, opening_prices, highest_prices, \
        lowest_prices, close_prices = \
        np.loadtxt(filename,
                   delimiter=',',
                   usecols=(1, 3, 4, 5, 6),
                   unpack=True,
                   converters={1: dmy2weekday})
    return weekdays[:16], opening_prices[:16], highest_prices[:16], \
        lowest_prices[:16], close_prices[:16]


def calc_average_prices(weekdays, close_prices):
    average_prices = np.zeros(5)
    for weekday in range(5):
        average_prices[weekday] = np.take(close_prices,
                                          np.where(weekdays == weekday)).mean()
    return average_prices


def get_summary(week_index, opening_prices,
                highest_prices, lowest_prices, close_prices):
    opening_price = np.take(opening_prices, week_index)[0]
    highest_price = np.max(np.take(highest_prices, week_index))
    lowest_price = np.min(np.take(lowest_prices, week_index))
    closeing_price = np.take(close_prices, week_index)[-1]
    return opening_price, highest_price, lowest_price, closeing_price


def savefile(filename, summary):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for summ in summary:
            row = list(summ)
            row.insert(0, 'AAPL')
            writer.writerow(row)


def main():
    week_days, opening_prices, highest_prices, \
        lowest_prices, close_prices = loadtxt('./data/data/aapl.csv')
    # print(average_prices)
    # print(week_days, opening_prices, highest_prices,
    # lowest_prices, close_prices)
    first = np.where(week_days == 0)[0][0]
    last = np.where(week_days == 4)[0][-1]
    week_index = np.arange(first, last + 1)
    week_index = np.split(week_index, 3)
    # print(week_index)
    # s_week = ('MON', 'TUE', 'WEN', 'THU', 'FRI')
    summary = np.apply_along_axis(get_summary, 1, week_index,
                                  opening_prices, highest_prices,
                                  lowest_prices, close_prices)
    print(summary)
    savefile('ppal_summary.csv', summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
