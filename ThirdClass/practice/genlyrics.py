# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'genlyrics.py'
@Time: '2020/4/7 20:05'
@Desc : ''
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as tfk

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("epochs")
    plt.ylabel(string)
    plt.show()

if __name__ == '__main__':
    tokenizer = tfk.preprocessing.text.Tokenizer()

    data = open('./irish-lyrics-eof.txt').read()

    corpus = data.lower().split('\n')

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

    xs, labels = input_sequences[:, :-1], input_sequences[:, -1]

    ys = tfk.utils.to_categorical(labels, num_classes=total_words)

    model = tfk.Sequential([
        tfk.layers.Embedding(total_words, 100, input_length=max_sequences_len-1),
        tfk.layers.Bidirectional(tfk.layers.LSTM(150)),
        tfk.layers.Dense(total_words, activation='softmax')
    ])

    adam = tfk.optimizers.Adam(lr=0.01)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

    history = model.fit(xs, ys, epochs=100, verbose=1)

    plot_graphs(history, 'acc')

    seed_text = "I've got a bad feeling about this"
    next_words = 100

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tfk.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequences_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        outout_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                outout_word = word
                break

        seed_text += " " + outout_word

    print(seed_text)