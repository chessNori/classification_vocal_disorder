import load_data
import time
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Conv2D, Flatten
from tensorflow.keras import Model

batch_size = 32
lr = 1e-6
EPOCHS = 300

start = time.time()
train_data = load_data.Data(path_name='train')
test_data = load_data.Data(path_name='test')

x_data, y_data = train_data.load_data(5)
x_data_test, y_data_test = test_data.load_data(5)


print("Data Loading is Done! (", time.time() - start, ")")
print('Shape of train data(x,y):', x_data.shape, y_data.shape)
print('Shape of test data(x,y):', x_data_test.shape, y_data_test.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).shuffle(5000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_data_test, y_data_test)).batch(batch_size)


class MyModel(Model):
    def __init__(self, number_disorder):
        super(MyModel, self).__init__()
        self.gru1 = GRU(256, dropout=0.05, return_sequences=True, go_backwards=True)
        self.gru2 = GRU(256, dropout=0.05, return_sequences=True, go_backwards=True)
        self.d1 = TimeDistributed(Dense(128, activation='relu'))
        self.d2 = TimeDistributed(Dense(64, activation='relu'))
        self.d3 = TimeDistributed(Dense(32, activation='relu'))
        self.d4 = GRU(16)
        self.d_output = Dense(number_disorder, activation='softmax')

    def call(self, inputs):
        x = self.gru1(inputs)
        x = self.gru2(inputs)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d_output(x)
        return x


_model = MyModel(len(train_data.y_data))

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


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

    return tf.argmax(pred, axis=-1)


for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()

    for _x, _y in train_dataset:
        train_step(_x, _y)

    conf_test = tf.zeros((len(train_data.y_data), len(train_data.y_data)), dtype=tf.dtypes.int32)
    for _x, _y in test_dataset:
        predictions = test_step(_x, _y)
        conf_test += tf.math.confusion_matrix(_y, predictions, num_classes=len(train_data.y_data))

    print(
        f'Epoch {epoch + 1}, '
        f'Train Loss: {train_loss.result()}, '
        f'Train Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}, '
        f'Time: {time.time() - start} sec'
    )
    print("Confusion matrix[test]\n", conf_test)
