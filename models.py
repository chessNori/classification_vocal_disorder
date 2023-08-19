import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Conv2D, Flatten, Concatenate, MaxPool2D, PReLU
from tensorflow.keras import Model


class Vox(Model):
    def __init__(self, number_disorder, win_size):
        super(Vox, self).__init__()
        self.win_size = 1024
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


class Capstone(Model):
    def __init__(self, number_disorder):
        super(Capstone, self).__init__()
        self.conv1_1 = Conv2D(8, 3, activation='relu', padding='same')
        self.conv1_2 = Conv2D(16, 3, activation='relu', padding='same')
        self.conv2_1 = Conv2D(8, 5, activation='relu', padding='same')
        self.conv2_2 = Conv2D(16, 5, activation='relu', padding='same')
        self.con = Concatenate(axis=-1)
        self.max_pool = MaxPool2D(pool_size=(7, 7), strides=(5, 1))
        self.conv4 = Conv2D(1, 5, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(256)
        self.PReLU1 = PReLU()
        self.d2 = Dense(128)
        self.PReLU2 = PReLU()
        self.output_unit = Dense(number_disorder, activation='softmax')

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)
        x = self.con([x1, x2])
        x = self.max_pool(x)
        x = self.conv4(x)  # (200, 200, 14, 1)
        x = self.flatten(x)  # (200, 2800)
        x = self.d1(x)
        x = self.PReLU1(x)
        x = self.d2(x)
        x = self.PReLU2(x)
        x = self.output_unit(x)

        return x
