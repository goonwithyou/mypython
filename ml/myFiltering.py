# @Time    : 2018/6/20 22:44
# @Author  : cap
# @FileName: myFiltering.py
# @Software: PyCharm Community Edition
# @introduction: make 滤波,傅里叶变换

import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as mp


def noised_signals(duration, sample_rate, signal_freq):
    """
    :param duration: 持续时间
    :param sample_rate: 采样频率
    :param signal_freq: 信号频率Hz
    :return:
    """
    times = np.linspace(0, duration, duration * sample_rate)
    # 滤波加噪声ω = 2πf = 2π/T
    noised_sig = np.sin(2 * np.pi * signal_freq * times) + np.random.randn(times.size) / 2
    return times, noised_sig


def save_signals(filename, sample_rate, sigs):
    sigs = sigs / sigs.max() * 2 ** 15
    wf.write(filename, sample_rate, sigs.astype(np.int16))


def init_noised_signals():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(221)
    mp.title('Noised Signal', fontsize=20)
    mp.xlabel('Time(ms)', fontsize=14)
    mp.ylabel('Signal', fontsize=14)
    mp.tick_params(which='both', top=True, right='True', labelright='True', labelsize=10)
    mp.grid(linestyle=':')


def draw_noised_signals(times, noised_sigs):
    times, noised_sigs = times[:200] * 1000, noised_sigs[: 200]
    mp.plot(times, noised_sigs, c='orangered', label='Nosied', zorder=0)
    mp.scatter(times, noised_sigs, edgecolors='orangered',
               facecolor='white', label='Sample', zorder=1)
    mp.legend()


def show():
    mp.show()


def main():
    DURATION = 5
    SAMPLE_RATE = 44100
    SIGNAL_FREQ = 1000
    times, noised_sig = noised_signals(DURATION, SAMPLE_RATE, SIGNAL_FREQ)
    # save_signals('noised.wav', SAMPLE_RATE, noised_sig)
    init_noised_signals()
    draw_noised_signals(times, noised_sig)
    show()

if __name__ == '__main__':
    main()
