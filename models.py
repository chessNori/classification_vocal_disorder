import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Conv2D, Flatten, Concatenate, MaxPool2D, PReLU
from tensorflow.keras import Model


class Vox(Model):
    def __init__(self, number_disorder, win_size):
        super(Vox, self).__init__()
        self.win_size = 512
        self.gru1 = GRU(256, dropout=0.05, return_sequences=True, go_backwards=True)
        self.d1 = TimeDistributed(Dense(128, activation='relu'))
        self.d2 = TimeDistributed(Dense(64, activation='relu'))
        self.d3 = TimeDistributed(Dense(32, activation='relu'))
        self.gru2 = GRU(16)
        self.d_output = Dense(number_disorder, activation='softmax')

    def call(self, inputs):
        x = tf.signal.stft(inputs, frame_length=self.win_size, frame_step=self.win_size//2, fft_length=self.win_size)
        x = tf.math.abs(x)
        x = tf.math.add(x, 1e-3)
        x = tf.math.log(x)
        x = tf.math.add(x, 3.)
        x = self.gru1(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.gru2(x)
        x = self.d_output(x)
        return x
