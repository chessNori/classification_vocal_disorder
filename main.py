import load_data
import time
import tensorflow as tf

batch_size = 8
lr = 1e-6

start = time.time()
train_data = load_data.Data(path_name='train')
test_data = load_data.Data(path_name='test')

x_data, y_data = train_data.load_data(3)
x_data_test, y_data_test = test_data.load_data(2)


print("Data Loading is Done! (", time.time() - start, ")")
print('Shape of train data(x,y):', x_data.shape, y_data.shape)
print('Shape of test data(x,y):', x_data_test.shape, y_data_test.shape)

# train_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).shuffle(5000).batch(batch_size)
# test_dataset = tf.data.Dataset.from_tensor_slices((x_data_test, y_data_test)).batch(batch_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(66560,), name='input'),
    tf.keras.layers.Lambda(lambda x: tf.signal.stft(x, frame_length=512, frame_step=256, fft_length=512)),
    tf.keras.layers.Lambda(lambda x: tf.math.abs(x)),
    tf.keras.layers.Lambda(lambda x: tf.math.add(x, 1e-3)),
    tf.keras.layers.Lambda(lambda x: tf.math.log(x)),
    tf.keras.layers.Lambda(lambda x: tf.math.add(x, 3.)),
    tf.keras.layers.GRU(256, dropout=0.05, return_sequences=True, go_backwards=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
    tf.keras.layers.GRU(16),
    tf.keras.layers.Dense(4, activation='softmax', name='output')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'))

model.fit(x_data, y_data, batch_size=batch_size, epochs=30, validation_data=(x_data_test, y_data_test))
pred = tf.argmax(model(x_data_test), axis=-1)
conf = tf.zeros((4, 4), dtype=tf.dtypes.int32)
conf += tf.math.confusion_matrix(y_data_test, pred, num_classes=4)
print(conf)
print('Normal, Papilloma, Paralysis, Vox senilis')
