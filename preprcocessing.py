# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:36:59 2018

@author: spinbjy
"""

import scipy.io.wavfile as sio_wav
import librosa
import numpy as np
import matplotlib.pyplot as plt

def PlotWaveform(audiofile):
    _,audiodata = sio_wav.read(audiofile)
    length = audiodata.shape[0]
    k = np.linspace(0,length-1,length)
    plt.plot(k,audiodata)
    plt.show()
    
def PlotSpectrum(audiofile):
    rate,audiodata = sio_wav.read(audiofile)
    Fdata = np.fft.rfft(audiodata) / rate
    freqs = np.linspace(0,rate//2,rate//2+1)
    for i in range(Fdata.shape[0]):
        if Fdata[i] == 0:
            print('Amp = 0')
            raise ValueError
    FdataLog = 20*np.log10(np.abs(Fdata))
    plt.plot(freqs,FdataLog)
    plt.show()
    
def PreEmphasis(audiofile):
    '''
        PreEmphasis is a High-pass filter to emphasis the high
        frequency in the audio. The transfer function is
        H(z) = 1 - az^(-1). So y[n] = x[n] - ax[n-1].
        We choose a as 0.975.
    '''
    a = 0.975
    sampling_rate,audiodata = sio_wav.read(audiofile)
    for i in range(audiodata.shape[0]):
        if i == 0:
            audiodata[i] = audiodata[i]
        else:
            audiodata[i] = audiodata[i] - a * audiodata[i - 1]
    return sampling_rate,audiodata

def MFCC(audiodata,sr,n_mfcc,n_fft,hop_length):
    audiodata = audiodata.astype('float32')
    return librosa.feature.mfcc(audiodata,sr,n_mfcc=n_mfcc,n_fft=n_fft,hop_length=hop_length)
    

if __name__ == '__main__':
    audiofile = 'test.wav'
    sr,audiodata = PreEmphasis(audiofile)
    mfcc = MFCC(audiodata,sr,20,400,240)
    