# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'practice.py'
@Time: '2020/3/30 16:32'
@Desc : '简单自然语言处理练习'
'''

import tensorflow as tf
import tensorflow.keras as tfk

if __name__ == '__main__':
    sentences = [
        'i love my dog',
        'I, love my cat',
        'You love my dog!',
        'Do you think my dog is amazing?'
    ]

    # token生成器，从文本中生成字典并创建向量，100为出现频率从大到小排列后的前100个单词生成向量
    # oov_token参数设置的是如果数据中出现没见过的单词时的填充符号
    tokenizer = tfk.preprocessing.text.Tokenizer(oov_token="<OOV>")
    # 接收文本数据并编码
    tokenizer.fit_on_texts(sentences)
    # 根据输入文本生成的语料库字典
    word_index = tokenizer.word_index
    # 将输入的文本按上一步生成的语料库重新编码为词向量
    sequences = tokenizer.texts_to_sequences(sentences)
    # 对短句子进行填充，使得所有句子长度一致，结果为一个矩阵，（句子数量，最长句子的列数），填充值为0
    # maxlen可以限制列数，但可能会造成长句子信息丢失
    # truncating与maxlen一起使用，作用是删除超过长度的部分，值为pre，post，默认pre即从前至后删除，post为从后向前
    # padding为填充方向，默认在句子前填充，即短句子前半部分为填充部分，为0，post为后半填充，即短句子后半为0
    padded = tfk.preprocessing.sequence.pad_sequences(sequences, padding='post', maxlen=5, truncating='post')

    print(word_index)
    print(sequences)
    print(padded)

    test_data = [
        'i really love my dog',
        'my dog loves my manatee'
    ]

    test_seq = tokenizer.texts_to_sequences(test_data)
    print(test_seq)