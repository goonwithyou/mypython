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
    :param signal_freq: 信号频率(强度)Hz(能量)
    :return:
    """
    times = np.linspace(0, duration, duration * sample_rate)
    # 滤波加噪声ω = 2πf = 2π/T，sin(wx)
    noised_sig = np.sin(2 * np.pi * signal_freq * times) + np.random.randn(times.size) / 2
    return times, noised_sig


def save_signals(filename, sample_rate, sigs):
    sigs = sigs / sigs.max() * 2 ** 15
    wf.write(filename, sample_rate, sigs.astype(np.int16))


# 时间转频率
def time2freq(sigs, sample_rate):
    # 傅里叶变换频率
    freqs = nf.fftfreq(sigs.size, d=1/sample_rate)
    #
    ffts = nf.fft(sigs)
    #
    amps = np.abs(ffts)
    return freqs, ffts, amps


def filter(freqs, noised_ffts, noised_amps):
    # 基音频率
    fund_freq = np.abs(freqs[noised_amps.argmax()])
    hight_freqs = np.where(np.abs(freqs) != fund_freq)
    filtered_ffts = noised_ffts.copy()
    filtered_ffts[hight_freqs] = 0
    filtered_amps = noised_amps.copy()
    filtered_amps[hight_freqs] = 0
    return fund_freq, filtered_ffts, filtered_amps


def freq2time(ffts):
    sigs = nf.ifft(ffts)
    return sigs.real


def init_noised_signals():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(221)
    mp.title('Noised Signal', fontsize=16)
    mp.xlabel('Time(ms)', fontsize=12)
    mp.ylabel('Signal', fontsize=12)
    mp.tick_params(which='both', top=True, right='True', labelright='True', labelsize=10)
    mp.grid(linestyle=':')


def draw_noised_signals(times, noised_sigs):
    times, noised_sigs = times[:200] * 1000, noised_sigs[: 200]
    mp.plot(times, noised_sigs, c='orangered', label='Nosied', zorder=0)
    mp.scatter(times, noised_sigs, edgecolors='orangered',
               facecolor='white', label='Sample', zorder=1)
    mp.legend()


def init_noised_amplitudes():
    mp.subplot(222)
    mp.title('Noised Amplitude', fontsize=16)
    mp.xlabel('Frequency(kHz)', fontsize = 12)
    mp.ylabel('Amplitude', fontsize=12)
    mp.tick_params(which='both', top=True, right=True, labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_noised_amplitudes(freqs, noised_amps):
    noised_amps = noised_amps[freqs > 0]
    freqs = freqs[freqs > 0] / 1000
    mp.semilogy(freqs, noised_amps, c='limegreen', label='Noised')
    mp.legend()


def init_filter_signals():
    mp.subplot(223)
    mp.title('Filter Signal', fontsize=16)
    mp.xlabel('Time(ms)', fontsize=12)
    mp.ylabel('Signal', fontsize=12)
    mp.tick_params(which='both', top=True, right='True', labelright='True', labelsize=10)
    mp.grid(linestyle=':')


def draw_filter_signals(times, filter_sigs):
    times, filter_sigs = times[:200] * 1000, filter_sigs[: 200]
    mp.plot(times, filter_sigs, c='orangered', label='Nosied', zorder=0)
    mp.scatter(times, filter_sigs, edgecolors='orangered',
               facecolor='white', label='Sample', zorder=0)
    mp.legend()


def init_filtered_amplitudes():
    mp.subplot(224)
    mp.title('Filtered Amplitude', fontsize=16)
    mp.xlabel('Frequency(kHz)', fontsize = 12)
    mp.ylabel('Amplitude', fontsize=12)
    mp.tick_params(which='both', top=True, right=True, labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_filtered_amplitudes(freqs, noised_amps):
    noised_amps = noised_amps[freqs > 0] + 1
    freqs = freqs[freqs > 0] / 1000
    mp.semilogy(freqs, noised_amps, c='limegreen', label='Noised')
    mp.legend()


def show():
    mp.show()


def main():
    DURATION = 5
    SAMPLE_RATE = 44100
    SIGNAL_FREQ = 1000
    times, noised_sig = noised_signals(DURATION, SAMPLE_RATE, SIGNAL_FREQ)
    freqs, noised_ffts, noised_amps = time2freq(noised_sig, SAMPLE_RATE)
    fund_freq, filtered_ffts, filtered_amps = filter(freqs, noised_ffts, noised_amps)
    filter_sigs = freq2time(filtered_ffts)
    # save_signals('noised.wav', SAMPLE_RATE, noised_sig)
    save_signals('filter.wav', SAMPLE_RATE, filter_sigs)
    init_noised_signals()
    draw_noised_signals(times, noised_sig)
    init_noised_amplitudes()
    draw_noised_amplitudes(freqs, noised_amps)
    init_filtered_amplitudes()
    draw_filtered_amplitudes(freqs, filtered_amps)
    init_filter_signals()
    draw_filter_signals(times, filter_sigs)
    show()

if __name__ == '__main__':
    main()
