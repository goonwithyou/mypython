# @Time    : 2018/6/18 17:06
# @Author  : cap
# @FileName: mySignal.py
# @Software: PyCharm Community Edition
# @introduction: 读取音频文件

import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as mp


def show_signal(sigs, sample_rate):
    print(sigs.shape)
    print(sample_rate / 1000, 'KHz')
    print(len(sigs) / sample_rate, 's')


def read_signals(filename):
    sample_rate, sigs = wf.read(filename)
    show_signal(sigs, sample_rate)
    # 选择前三十个点
    sigs = sigs[: 30] /2 ** 15
    times = np.arange(30) / sample_rate
    return sigs, sample_rate, times


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('Audio Signal', fontsize=20)
    mp.xlabel('Time (ms)', fontsize=14)
    mp.ylabel('Signal', fontsize=14)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_chart(times, sigs):
    times *= 1000
    mp.plot(times, sigs, c='dodgerblue', label='Signal', zorder=0)
    mp.scatter(times, sigs, edgecolor='orangered',
               facecolor='white', s=80, label='Sample', zorder=1)
    mp.legend()


def show_chart():
    mp.show()


def main():
    sigs, sample_rate, times = read_signals('./data/signal.wav')
    init_chart()
    draw_chart(times, sigs)
    show_chart()


if __name__ == '__main__':
    main()
