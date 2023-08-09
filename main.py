import numpy as np
import load_data
import time
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, TimeDistributed
from tensorflow.keras import Model

batch_size = 16
lr = 1e-4
EPOCHS = 150

start = time.time()
data = load_data.Data(500)

x_data, y_data = data.load_data(0)

for i in range(1, 3):
    x_temp, y_temp = data.load_data(i)
    x_data = np.concatenate((x_data, x_temp), axis=0)
    y_data = np.concatenate((y_data, y_temp), axis=0)

data_test = load_data.Data(100)

x_data_test, y_data_test = data_test.load_data(0)

for i in range(1, 3):
    x_temp, y_temp = data_test.load_data(i)
    x_data_test = np.concatenate((x_data_test, x_temp), axis=0)
    y_data_test = np.concatenate((y_data_test, y_temp), axis=0)


print("Data Loading is Done! (", time.time() - start, ")")
print('Shape of train data(x,y):', x_data.shape, y_data.shape)
print('Shape of test data(x,y):', x_data_test.shape, y_data_test.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).shuffle(1500).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_data_test, y_data_test)).batch(batch_size)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.gru = GRU(200, return_sequences=True)
        self.d1 = TimeDistributed(Dense(100))
        self.d2 = TimeDistributed(Dense(3, activation='softmax'))

    def call(self, inputs):
        x = self.gru(inputs)
        x = self.d1(x)
        x = self.d2(x)

        return x


_model = MyModel()

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_object = tf.keras.losses.CategoricalCrossentropy()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(wave, index):
    with tf.GradientTape() as tape:
        pred = _model(wave, training=True)
        loss = loss_object(index, pred)
    gradients = tape.gradient(loss, _model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, _model.trainable_variables))

    train_loss(loss)
    train_accuracy(index, pred)


@tf.function
def test_step(wave, index):
    pred = _model(wave, training=False)
    loss = loss_object(index, pred)

    test_loss(loss)
    test_accuracy(index, pred)


for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()

    for _x, _y in train_dataset:
        train_step(_x, _y)

    for _x, _y in test_dataset:
        test_step(_x, _y)

    print(
        f'Epoch {epoch + 1}, '
        f'Train Loss: {train_loss.result()}, '
        f'Train Accuracy: {train_accuracy.result()}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result()}, '
        f'Time: {time.time() - start} sec'
    )
