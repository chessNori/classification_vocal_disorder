import load_data
import time
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Conv2D, Flatten
from tensorflow.keras import Model

batch_size = 16
lr = 1e-5
EPOCHS = 150

start = time.time()
train_data = load_data.Data(path_name='train')
test_data = load_data.Data(path_name='test')

x_data, y_data = train_data.load_data(5)
x_data_test, y_data_test = test_data.load_data(5)


print("Data Loading is Done! (", time.time() - start, ")")
print('Shape of train data(x,y):', x_data.shape, y_data.shape)
print('Shape of test data(x,y):', x_data_test.shape, y_data_test.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).shuffle(1400).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_data_test, y_data_test)).batch(batch_size)


class MyModel(Model):
    def __init__(self, number_disorder):
        super(MyModel, self).__init__()
        self.gru = GRU(300, return_sequences=True)
        self.d1 = TimeDistributed(Dense(150))
        self.d2 = TimeDistributed(Dense(number_disorder, activation='softmax'))

    def call(self, inputs):
        x = self.gru(inputs)
        x = self.d1(x)
        x = self.d2(x)

        return x


_model = MyModel(len(train_data.y_data))

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
        _index = tf.cast(_y, tf.int32)
        _index = tf.one_hot(_index, len(train_data.y_data), axis=-1)
        train_step(_x, _index)

    for _x, _y in test_dataset:
        _index = tf.cast(_y, tf.int32)
        _index = tf.one_hot(_index, len(train_data.y_data), axis=-1)
        test_step(_x, _index)

    print(
        f'Epoch {epoch + 1}, '
        f'Train Loss: {train_loss.result()}, '
        f'Train Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}, '
        f'Time: {time.time() - start} sec'
    )
