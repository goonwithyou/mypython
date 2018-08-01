import os
import sys
import platform
import datetime as dt
import numpy as np
import matplotlib.pyplot as mp
import matplotlib.dates as md


def dmy2ymd(dmy):
    return dt.datetime.strptime(str(dmy, encoding='utf-8'),
                                '%d-%m-%Y').date().strftime('%Y-%m-%d')


def read_data(filename):
    dates, opening_price, highest_price,\
        lowest_price, close_price = np.loadtxt(
            filename,
            delimiter=',',
            usecols=(1, 3, 4, 5, 6),
            unpack=True,
            dtype=np.dtype('M8[D], f8, f8, f8, f8'),
            converters={1: dmy2ymd})
    # type of dates is python_datetime

    # print(dates)
    return dates, opening_price, highest_price,\
        lowest_price, close_price


def init_chart(first_day, last_day):
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('Candlestick Chart', fontsize=20)
    mp.xlabel('Trading Days From %s To %s' % (
        first_day.astype(md.datetime.datetime).strftime('%d-%m-%Y'),
        last_day.astype(md.datetime.datetime).strftime('%d-%m-%Y')),
        fontsize=14)
    mp.ylabel('Stock Price (USD) Of Apple Inc.', fontsize=14)

    # 设置刻度
    # get current axes
    ax = mp.gca()
    # 设置主刻度的显示格式，（周，周一）
    ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=md.MO))
    # 设置副刻度（天）
    ax.xaxis.set_minor_locator(md.DayLocator())
    # 设置主刻度的显示格式
    ax.xaxis.set_major_formatter(md.DateFormatter('%d-%m-%Y'))

    # 设置刻度的显示位置
    mp.tick_params(which='both',
                   top=True,
                   right=True,
                   labelright=True,
                   labelsize=10)
    mp.grid(linestyle=':')


def draw_chart(dates, opening_price, highest_price,
               lowest_price, close_price):
    # reset dates type to md.datetime.datetime
    dates = dates.astype(md.datetime.datetime)
    up = close_price - opening_price >= 1e-2
    down = opening_price - close_price >= 1e-2

    # 上涨的为红边,白芯．下跌的为绿边绿芯．不涨不跌的为黑
    # 填充色，每个元素三个浮点数（r, g, b），４字节
    fc = np.zeros(dates.size, dtype='3f4')
    # 上涨的填充色设为白色，下跌的填充色设为绿色
    fc[up], fc[down] = (1, 1, 1), (0, 0.5, 0)
    # 边缘色
    ec = np.zeros(dates.size, dtype='3f4')
    ec[up], ec[down] = (1, 0, 0), (0, 0.5, 0)

    # mp.bar(水平坐标，高度，比例，起始值)
    # 影线
    mp.bar(dates, highest_price - lowest_price, 0,
           lowest_price, align='center', color=fc, edgecolor=ec)
    # 实体
    mp.bar(dates, close_price - opening_price, 0.8,
           opening_price, align='center', color=fc, edgecolor=ec)

    mp.gcf().autofmt_xdate()


def show_chart():
    # setting biggest width and height of window
    # mng = mp.get_current_fig_manager()
    # if 'Windows' in platform.system():
    #     mng.window.state()
    # else:
    #     size = mng.window.maximumSize()
    #     width = size.width()
    #     height = size.height()
    #     mng.resize(width, height)
    mp.show()


def main(argc, argv, envp):
    dates, opening_price, highest_price,\
        lowest_price, close_price = read_data('./data/data/aapl.csv')
    # print(dates.dtype)
    init_chart(dates[0], dates[-1])
    draw_chart(dates, opening_price, highest_price,
               lowest_price, close_price)
    show_chart()
    return 0


if __name__ == '__main__':
    # main(len(sys.argv), sys.argv, os.environ)
    sys.exit(main(len(sys.argv), sys.argv, os.environ))