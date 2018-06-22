# @Time    : 2018/6/18 21:24
# @Author  : cap
# @FileName: myFreq.py
# @Software: PyCharm Community Edition
# @introduction: 基于傅里叶变换的频率域音频数据

import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as mp
import numpy.fft as nf
# 快速傅里叶


def show_signal(sigs, sample_rate):
    print(sigs.shape)
    print(sample_rate / 1000, 'KHz')
    print(len(sigs) / sample_rate, 's')


def read_signals(filename):
    sample_rate, sigs = wf.read(filename)
    show_signal(sigs, sample_rate)
    # 选择前三十个点
    sigs = sigs / 2 ** 15
    times = np.arange(len(sigs)) / sample_rate
    return sigs, sample_rate, times


def time2freq(sigs, sample_rate):
    # 频率hz，生成频率间隔为sample_rate/len(sigs)的间隔的长度为len(sigs)的等差频率数组
    freqs = nf.fftfreq(len(sigs), d= 1 / sample_rate)
    # 复数表示的x, y
    ffts = nf.fft(sigs)
    # 斜边长度，声音的强度
    amps = np.abs(ffts)
    return freqs, ffts, amps


def init_signals():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(121)
    mp.title('Audio Signal', fontsize=20)
    mp.xlabel('Time (ms)', fontsize=14)
    mp.ylabel('Signal', fontsize=14)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_signals(times, sigs):
    times *= 1000
    mp.plot(times, sigs, c='dodgerblue', label='Signal', zorder=0)
    # mp.scatter(times, sigs, edgecolor='orangered',
    #            facecolor='white', s=80, label='Sample', zorder=1)
    mp.legend()


def init_amplitudes():
    mp.subplot(122)
    mp.title('Audio Amplitude', fontsize=20)
    mp.xlabel('Frequency (kHz)', fontsize=14)
    mp.ylabel('Amplitude', fontsize=14)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_amplitudes(freqs, amps):
    amps = amps[freqs >= 0]
    freqs = freqs[freqs >= 0] /1000
    mp.semilogy(freqs, amps, c='orangered', label='Amplitude')
    # mp.plot(freqs, amps, c='orangered', label='Amplitude')
    mp.legend()

def show_chart():
    mp.show()


def main():
    sigs, sample_rate, times = read_signals('./data/freq.wav')
    init_signals()
    draw_signals(times, sigs)

    freqs, ffts, amps = time2freq(sigs, sample_rate)
    init_amplitudes()
    draw_amplitudes(freqs, amps)
    show_chart()


if __name__ == '__main__':
    main()
