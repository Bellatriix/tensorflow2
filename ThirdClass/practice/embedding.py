# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'embedding.py'
@Time: '2020/3/30 21:05'
@Desc : '使用IMDB影评练习词嵌入'
'''

import io
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_datasets as tfds

if __name__ == '__main__':
    # 加载imdb数据
    imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

    train_data, test_data = imdb['train'], imdb['test']

    train_sentences = []
    train_labels = []

    test_sentences = []
    test_labels = []

    for s,l in train_data:
        train_sentences.append(str(s.numpy()))
        train_labels.append(l.numpy())

    for s,l in test_data:
        test_sentences.append(str(s.numpy()))
        test_labels.append(l.numpy())

    train_labels_final = np.array(train_labels)
    test_labels_final = np.array(test_labels)

    # 标记句子
    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    oov_tok = '<OOV>'

    tokenizer = tfk.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(train_sentences)
    padded = tfk.preprocessing.sequence.pad_sequences(sequences,
                                                      maxlen=max_length,
                                                      truncating=trunc_type)

    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_padded = tfk.preprocessing.sequence.pad_sequences(test_sequences,
                                                           maxlen=max_length)

    # 建立神经网络
    model = tfk.Sequential([
        tfk.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tfk.layers.Flatten(),
        tfk.layers.Dense(6, activation='relu'),
        tfk.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(padded, train_labels_final, epochs=10, validation_data=(test_padded, test_labels_final))

    e = model.layers[0]
    weights = e.get_weights()[0]
    print(weights.shape)

    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    for word_num in range(1, vocab_size):
        word = reverse_word_index[word_num]
        embeddings = weights[word_num]
        out_m.write(word+'\n')
        out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    out_m.close()
    out_v.close()