# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'CatVSDog.py'
@Time: '2020/3/26 15:07'
@Desc : '简单卷积神经网络识别猫和狗'
'''

import tensorflow as tf
import tensorflow.keras as tfk

if __name__ == '__main__':
    model = tfk.Sequential([
        tfk.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Conv2D(32, (3, 3), activation='relu'),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Conv2D(64, (3, 3), activation='relu'),
        tfk.layers.MaxPooling2D(2, 2),
        tfk.layers.Flatten(),
        tfk.layers.Dense(512, activation='relu'),
        tfk.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=tfk.optimizers.RMSprop(lr=0.001), metrics=['accuracy'])

    train_datagen = tfk.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    test_datagen = tfk.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        './train',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )
    test_generator = test_datagen.flow_from_directory(
        './validation',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    model.fit(train_generator, validation_data=test_generator, steps_per_epoch=100,
              epochs=15, validation_steps=50)
