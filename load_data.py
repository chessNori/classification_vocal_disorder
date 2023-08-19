import librosa
import numpy as np
import os
from glob import glob
import random
import copy


class Data:
    def __init__(self, resampling=16000, n_fft=512, win_size=512, path_name=None):
        if path_name is None:
            self.path = '..\\datasets\\original_data\\patient-vocal-dataset\\patient-vocal-dataset\\'
        else:
            self.path = '..\\datasets\\' + path_name + '\\'
        self.sr = resampling  # We use 8k sampling datasets
        self.n_fft = n_fft  # FFT N value
        self.win_size = win_size
        self.frame_num = 260  # why?
        self.padding = n_fft * self.frame_num  # Output results data size for regularization

        disorder_dir = [f.path for f in os.scandir(self.path) if f.is_dir()]
        self.data = []
        for i in range(len(disorder_dir)):
            self.data += [[]]  # Memory for dynamic

        wave_file_name = []
        for one_path in disorder_dir:
            wave_file_name += glob(one_path + '\\*.wav')
            wave_file_name += glob(one_path + '\\*.mp3')

        delete_file = []
        for one_path in wave_file_name:
            if len(one_path.split('egg')) > 1:
                delete_file.append(one_path)  # Not using EGG signal data

        for one_path in delete_file:
            wave_file_name.remove(one_path)  # Delete EGG file

        temp = []
        for i in range(len(disorder_dir)):
            temp += [[]]

        for one_path in wave_file_name:
            point = one_path.split('\\')[-2]
            for i in range(len(disorder_dir)):
                target = disorder_dir[i].split('\\')[-1]
                if point == target:
                    temp[i].append(one_path)
                    break

        self.file_name = copy.deepcopy(temp)

        temp = []
        for i in range(len(disorder_dir)):
            temp += [[]]
            temp[i] = np.zeros((len(self.file_name[i])), dtype=np.int32) + i
        self.y_data = copy.deepcopy(temp)

    def rnn_shape(self, wave):  # ( 1, frame_num, N/2 )
        spectrum = librosa.stft(wave, n_fft=self.n_fft, hop_length=self.win_size // 2, win_length=self.win_size,
                                window='hann')[1:, :self.frame_num]  # (n_fft/2, frame_num)
        spectrum = np.transpose(spectrum, (1, 0))  # (frame_num, n_fft/2)
        spectrum = np.expand_dims(spectrum, axis=0)  # (1, frame_num, N/2)

        return spectrum

    def rnn_spectrogram(self, disorder_number, file_number):  # return magnitude
        print("Loading file_" + str(file_number) + ": ", self.file_name[disorder_number][file_number])
        wave, sr = librosa.load(self.file_name[disorder_number][file_number], sr=self.sr)
        if wave.shape[0] >= self.padding:
            wave = wave[:self.padding]
            print("The file size is bigger than padding size")
        else:
            wave = np.concatenate((wave, np.zeros(self.padding - wave.shape[0])), axis=0)

        spectrum = self.rnn_shape(wave)
        spectrum, _ = librosa.magphase(spectrum)
        # spectrum = np.log10(spectrum + 1e-9)
        spectrum.astype(np.float32)

        return spectrum

    def make_x_data(self):
        for i in range(len(self.file_name)):
            res = self.rnn_spectrogram(i, 0)
            val_max = np.max(res)
            # res /= val_max  # 0.0 ~ 1.0
            for j in range(1, len(self.file_name[i])):
                res_temp = self.rnn_spectrogram(i, j)
                val_max = np.max(res_temp)
                # res_temp /= val_max  # 0.0 ~ 1.0
                res = np.concatenate((res, res_temp), axis=0)
            self.data[i] = np.copy(res)

    def same_size_data(self, scale: int):
        number_data = []
        for i in range(len(self.data)):
            number_data.append(len(self.data[i]))

        target = max(number_data) * scale  # How many data in each label

        while True:
            for i in range(len(self.data)):
                if number_data[i] < target:
                    temp = np.copy(self.data[i])
                    temp *= random.random()
                    self.data[i] = np.concatenate((self.data[i], temp), axis=0)
                    self.y_data[i] = np.concatenate((self.y_data[i], self.y_data[i]), axis=0)
                    number_data[i] *= 2
                else:
                    self.data[i] = self.data[i][:target]
                    self.y_data[i] = self.y_data[i][:target]
                    number_data[i] = target

            cut = 1
            for i in range(len(number_data)):
                if number_data[i] != target:
                    cut *= 0
            if cut == 1:
                break

    def load_data(self, number_scale: int):
        self.make_x_data()
        self.same_size_data(number_scale)

        x = np.array(self.data)
        x = np.reshape(x, (-1, self.frame_num, self.n_fft//2))
        x.astype(np.float32)
        y = np.array(self.y_data)
        y = np.reshape(y, (-1))
        y.astype(np.int32)

        return x, y
