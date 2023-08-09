import librosa
import numpy as np
import os
from glob import glob
import random


class Data:
    def __init__(self, number_regularization, resampling=16000, n_fft=256, folder_name='train'):
        self.path = '..\\datasets\\' + folder_name + '\\'
        self.number_regularization = number_regularization
        self.sr = resampling
        self.n_fft = n_fft
        self.frame_num = 130
        self.padding = n_fft * self.frame_num

        disorder_dir = [f.path for f in os.scandir(self.path) if f.is_dir()]

        wave_file_name = []
        for one_path in disorder_dir:
            wave_file_name += glob(one_path + '\\*.wav')

        delete_file = []
        for one_path in wave_file_name:
            if len(one_path.split('egg')) > 1:
                delete_file.append(one_path)

        for one_path in delete_file:
            wave_file_name.remove(one_path)  # Delete EGG file
        delete_file = None
        disorder_dir = None

        file_name_laryngozele = []
        file_name_normal = []
        file_name_vox = []

        for one_path in wave_file_name:
            if one_path.split('\\')[-2][0] == 'L':  # First alphabet of disorder name
                file_name_laryngozele.append(one_path)
            elif one_path.split('\\')[-2][0] == 'N':
                file_name_normal.append(one_path)
            elif one_path.split('\\')[-2][0] == 'V':
                file_name_vox.append(one_path)
            else:
                print("ERROR: Check this file name ->", one_path)
                exit()

        temp = [None, None, None]
        temp[0] = file_name_laryngozele
        temp[1] = file_name_normal
        temp[2] = file_name_vox
        self.file_name = temp

        temp = None
        wave_file_name = None
        file_name_laryngozele = None
        file_name_normal = None
        file_name_vox = None

    def rnn_shape(self, wave):  # ( 1, frame_num, N // 2 + 1)
        spectrum = librosa.stft(wave, n_fft=self.n_fft, hop_length=self.n_fft // 2, win_length=self.n_fft,
                                window='hann')[:, :self.frame_num]  # (n_fft/2 + 1, frame_num * 2 + 1)
        spectrum = np.transpose(spectrum, (1, 0))
        spectrum = np.expand_dims(spectrum, axis=0)

        return spectrum

    def rnn_spectrogram(self, file_number, file_name):
        random_scale = random.random()
        print("Loading file_" + str(file_number) + ": ", file_name[file_number])
        print("Scale value:", random_scale)
        wave, sr = librosa.load(file_name[file_number], sr=self.sr)
        if wave.shape[0] >= self.padding:
            wave = wave[:self.padding]
            print("The file size is bigger than padding size")
        else:
            wave = np.concatenate((wave, np.zeros(self.padding - wave.shape[0])), axis=0)

        spectrum = self.rnn_shape(wave * random_scale)

        return spectrum

    def load_data(self, disorder_number):
        x_data = self.rnn_spectrogram(0, self.file_name[disorder_number])
        for i in range(1, len(self.file_name[disorder_number])):
            temp = self.rnn_spectrogram(i, self.file_name[disorder_number])
            x_data = np.concatenate((x_data, temp), axis=0)
        data_num = len(self.file_name[disorder_number])

        while data_num <= self.number_regularization:
            for i in range(len(self.file_name[disorder_number])):
                temp = self.rnn_spectrogram(i, self.file_name[disorder_number])
                x_data = np.concatenate((x_data, temp), axis=0)
            data_num += len(self.file_name[disorder_number])

        x_data = np.abs(x_data[:self.number_regularization])
        y_data = np.zeros((self.number_regularization, self.frame_num, 3))
        y_data[:, :, disorder_number] += 1

        return x_data, y_data

