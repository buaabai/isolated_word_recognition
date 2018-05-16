# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:36:59 2018

@author: spinbjy
"""

import scipy.io.wavfile as sio_wav
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import shutil

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
    stft = librosa.core.stft(audiodata,n_fft=n_fft,hop_length=hop_length,center=False)
    return librosa.feature.mfcc(audiodata,sr,n_mfcc=n_mfcc,S=stft,n_fft=n_fft,hop_length=hop_length)
    #return librosa.feature.mfcc(audiodata,sr,n_mfcc=n_mfcc,n_fft=n_fft,hop_length=hop_length)

def MfccOnDataset(datasetdir,mfccdatasetdir,n_mfcc,n_fft,hop_length):
    print('Extracting MFCC feature......')
    subdatasetlist = os.listdir(datasetdir)
    if not os.path.isdir(mfccdatasetdir):
        os.mkdir(mfccdatasetdir)
    for name in subdatasetlist:
        mfccsubdatadir = os.path.join(mfccdatasetdir,name)
        if not os.path.isdir(mfccsubdatadir):
            os.mkdir(mfccsubdatadir)
        subdatadir = os.path.join(datasetdir,name)
        audiofile = os.listdir(subdatadir)
        for file in audiofile:
            sr,audiodata = PreEmphasis(os.path.join(subdatadir,file))
            audio_mfcc = MFCC(audiodata,sr,n_mfcc,n_fft,hop_length)
            np.savetxt(os.path.join(mfccdatasetdir,name,file[:-4]+'.txt'),audio_mfcc)
    print('Extract MFCC features Done!')

def PreprocessOnMfccdataet(mfccdatasetdir,data_shape):
    print('Removing bad data whose shape is not equal to {}'.format(data_shape)+'......')
    sublist = os.listdir(mfccdatasetdir)
    for name in sublist:
        mfccsubdatadir = os.path.join(mfccdatasetdir,name)
        mfccfile = os.listdir(mfccsubdatadir)
        for file in mfccfile:
            filename = os.path.join(mfccsubdatadir,file)
            if np.loadtxt(filename).shape != data_shape:
                os.remove(filename)
    print('Remove bad data Done!')
    
def SmallMfccDataset(mfccdatasetdir,smallmfccdatasetdir,n_train,n_test):
    print('Start making a smaller mfcc dataset......')
    submfcclist = os.listdir(mfccdatasetdir)
    if not os.path.isdir(smallmfccdatasetdir):
        os.mkdir(smallmfccdatasetdir)
        
    train = os.path.join(smallmfccdatasetdir,'train')
    if not os.path.isdir(train):
        os.mkdir(train)
            
    test = os.path.join(smallmfccdatasetdir,'test')
    if not os.path.isdir(test):
        os.mkdir(test)
            
    for name in submfcclist:
        submfccdir = os.path.join(mfccdatasetdir,name)
        mfccdatafile = os.listdir(submfccdir)
        random.shuffle(mfccdatafile)
        
        train_sub = os.path.join(train,name)
        test_sub = os.path.join(test,name)
        if not os.path.isdir(train_sub):
            os.mkdir(train_sub)
        if not os.path.isdir(test_sub):
            os.mkdir(test_sub)
        
        for file in mfccdatafile[:n_train]:
            oldfile = os.path.join(submfccdir,file)
            newfile = os.path.join(train_sub,file)
            shutil.copyfile(oldfile,newfile)
        
        
        for file in mfccdatafile[n_train:n_train + n_test]:
            oldfile = os.path.join(submfccdir,file)
            newfile = os.path.join(test_sub,file)
            shutil.copyfile(oldfile,newfile)
    print('Extract dataset Done!')
            
            
if __name__ == '__main__':
    datasetdir = 'C:\\Users\\spinbjy\\Desktop\\test\\dataset'
    mfccdatadir = 'C:\\Users\\spinbjy\\Desktop\\test\\mfccdataset'
    smallmfccdir = 'C:\\Users\\spinbjy\\Desktop\\test\\smallmfcc'
    #MfccOnDataset(datasetdir,mfccdatadir,40,400,240)
    PreprocessOnMfccdataet(mfccdatadir,data_shape=(40,66))
    SmallMfccDataset(mfccdatadir,smallmfccdir,1800,200)


    