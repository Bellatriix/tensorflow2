# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'RNN+LSTM.py'
@Time: '2020/4/9 17:23'
@Desc : ''
'''

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 6 * np.pi),
                    2 / np.exp(9 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

if __name__ == '__main__':
    time = np.arange(10 * 365 + 1, dtype='float32')
    baseline = 10
    series = trend(time, 0.1)
    amplitude = 40
    slope = 0.005
    noise_level = 3

    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    series += noise(time, noise_level, seed=51)

    split_time = 3000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    window_size = 20
    batch_size = 32
    shuffle_buffer_size = 1000

    plot_series(time, series)

    tfk.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)

    dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

    model = tfk.Sequential([
        tfk.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        tfk.layers.Bidirectional(tfk.layers.LSTM(32, return_sequences=True)),
        tfk.layers.Bidirectional(tfk.layers.LSTM(32)),
        tfk.layers.Dense(1),
        tfk.layers.Lambda(lambda x: x * 10.0)
    ])

    # lr_schedule = tfk.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
    optimizer = tfk.optimizers.SGD(lr=1e-5, momentum=0.9)

    model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])

    history = model.fit(dataset, epochs=500, verbose=1)

    forecast = []
    results = []
    for time in range(len(series) - window_size):
        forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

    forecast = forecast[split_time-window_size:]
    results = np.array(forecast)[:, 0, 0]

    plt.figure(figsize=(10, 6))

    plot_series(time_valid, x_valid)
    plot_series(time_valid, results)

    print(tfk.metrics.mean_absolute_error(x_valid, results).numpy())

    mae = history.history['mae']
    loss = history.history['loss']

    epochs = range(len(loss))

    plt.plot(epochs, mae, 'r')
    plt.plot(epochs, loss, 'b')
    plt.title('MAE and Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["MAE", "Loss"])

    plt.figure()

    epochs_zoom = epochs[200:]
    mae_zoom = mae[200:]
    loss_zoom = loss[200:]

    plt.plot(epochs_zoom, mae_zoom, 'r')
    plt.plot(epochs_zoom, loss_zoom, 'b')
    plt.title('MAE and Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["MAE", "Loss"])

    plt.figure()
    plt.show()