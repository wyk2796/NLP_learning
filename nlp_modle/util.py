# coding:utf-8
import numpy as np


def one_hot(class_arr, class_num):
    return np.eye(len(class_arr), int(class_num))[class_arr.astype(dtype=np.int)]


def one_hot_simple(num, class_num):
    arr = np.zeros(shape=(1, class_num))
    arr[0][int(num)] = 1
    return arr


