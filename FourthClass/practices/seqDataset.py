# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'seqDataset.py'
@Time: '2020/4/8 17:07'
@Desc : 'tf建立简单序列数据集'
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''
@Desc : '根据输入数据生成数据窗口'
@Parameters :
    'series' - '输入数据集'
    'window_size' - '窗口大小'
    'batch_size' - '批次大小'
    'shuffle_buffer' - '乱序缓冲区大小'
@Returns :
    'dataset' - '修改后的数据集'
@Time : '2020/4/8 17:14'
'''
def window_dataset(series, window_size, batch_size, shuffle_buffer):
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices(series)
    # 将数据切分为窗口
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    # 将数据展平
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    # 将数据随机打乱
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    # 拆分数据和标签
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    # 划分为选定的批次大小并返回
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

if __name__ == '__main__':
    time = np.arange(4 * 365 + 1, dtype="float32")
    baseline = 10
    series = trend(time, 0.1)
    baseline = 10
    amplitude = 40
    slope = 0.05
    noise_level = 5

    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    series += noise(time, noise_level, seed=42)

    split_time = 1000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    window_size = 20
    batch_size = 32
    shuffle_buffer_size = 1000

    dataset = window_dataset(x_train, window_size, batch_size, shuffle_buffer_size)


    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=[window_size], activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # 通过回调找到最优情况下的学习率
    # lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    #     lambda epoch: 1e-8 * 10 ** (epoch / 20)
    # )

    model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=8e-6, momentum=0.9))
    history = model.fit(dataset, epochs=500)

    '''
    lrs = 1e-8 * (10 ** (np.arange(100) / 20))
    plt.semilogx(lrs, history.history["loss"])
    plt.axis([1e-8, 1e-3, 0, 300])
    plt.show()
    exit()
    '''

    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

    forecast = forecast[split_time-window_size:]
    results = np.array(forecast)[:, 0, 0]


    plt.figure(figsize=(10, 6))

    plot_series(time_valid, x_valid)
    plot_series(time_valid, results)
    plt.show()

    print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())