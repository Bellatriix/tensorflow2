# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'shakespeare.py'
@Time: '2020/4/7 20:55'
@Desc : 'RNN生成莎士比亚风格诗句'
'''

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = open('./sonnets.txt').read()

    corpus = data.lower().split('\n')

    tokenizer = tfk.preprocessing.text.Tokenizer()

    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequences = token_list[:i+1]
            input_sequences.append(n_gram_sequences)

    max_sequences_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(tfk.preprocessing.sequence.pad_sequences(input_sequences,
                                                                        maxlen=max_sequences_len,
                                                                        padding='pre'))

    predictors, labels = input_sequences[:, :-1], input_sequences[:, -1]
    labels = tfk.utils.to_categorical(labels, num_classes=total_words)

    model = tfk.Sequential([
        tfk.layers.Embedding(total_words, 100, input_length=max_sequences_len-1),
        tfk.layers.Bidirectional(tfk.layers.LSTM(150, return_sequences=True)),
        tfk.layers.Dropout(0.2),
        tfk.layers.LSTM(100),
        tfk.layers.Dense(total_words/2, activation="relu", kernel_regularizer=tfk.regularizers.l2(0.01)),
        tfk.layers.Dense(total_words, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    history = model.fit(predictors, labels, epochs=100, verbose=1)

    acc = history.history['acc']
    loss = history.history['loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.title('Training accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.legend()

    plt.show()

    seed_text = "Help me Obi Wan Kenobi, you're my only hope"
    next_words = 100

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tfk.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequences_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)