# coding:utf-8
import numpy as np

class GramNode:

    def __init__(self, target_words):
        self.target = target_words
        self.counter_map = {}
        self.counter = 0

    def add(self, after_word):
        if after_word in self.counter_map:
            self.counter_map[after_word] += 1
        else:
            self.counter_map[after_word] = 1
        self.counter += 1


class Ngram(object):

    def __init__(self, vocabulary_size):
        possibility_matrix = np.zeros([vocabulary_size,vocabulary_size],np.int)
        pass

    def calculater_ngram(self, content, gram_num=2):
        relative_map = {}
        for line in content:
            line




