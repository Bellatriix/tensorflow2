# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'news.py'
@Time: '2020/3/30 18:17'
@Desc : '新闻数据转变为词向量'
'''

import json
import tensorflow as tf
import tensorflow.keras as tfk

if __name__ == '__main__':
    with open('./sarcasm.json', 'r') as f:
        datastore = json.load(f)

    sentences = []
    labels = []
    urls = []

    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
        urls.append(item['article_link'])

    tonkenizer = tfk.preprocessing.text.Tokenizer(oov_token='<OOV>')
    tonkenizer.fit_on_texts(sentences)

    word_index = tonkenizer.word_index

    print(len(word_index))
    print(word_index)

    sequences = tonkenizer.texts_to_sequences(sentences)
    padded = tfk.preprocessing.sequence.pad_sequences(sequences, padding='post')

    print(padded[0])
    print(padded.shape)