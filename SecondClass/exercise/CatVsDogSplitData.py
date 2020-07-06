# anaconda/learn python
# -*- coding: utf-8 -*-

'''
@Author : '张鹏'
@Software : 'IntelliJ IDEA'
@File : 'CatVsDogSplitData.py'
@Time: '2020/3/26 15:53'
@Desc : '划分数据集，数据来源kaggle'
'''

import os
import random
from shutil import copyfile

'''
@Desc : '划分数据集'
@Parameters :
    'source' - '数据源地址'
    'training' - '训练集地址'
    'test' - '测试集地址'
    'split_size' - '划分比例'
@Returns : none
@Time : '2020/3/26 15:53'
'''
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)

    print('suc')

if __name__ == '__main__':
    CAT_SOURCE_DIR = "./PetImages/Cat/"
    TRAINING_CATS_DIR = "./cats-v-dogs/training/cats/"
    TESTING_CATS_DIR = "./cats-v-dogs/testing/cats/"
    DOG_SOURCE_DIR = "./PetImages/Dog/"
    TRAINING_DOGS_DIR = "./cats-v-dogs/training/dogs/"
    TESTING_DOGS_DIR = "./cats-v-dogs/testing/dogs/"

    split_size = .9
    split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
    split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)