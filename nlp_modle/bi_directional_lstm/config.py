# coding:utf-8


class Config(object):

    hidden_size = 200
    batch_size = 20
    corpus_vocabulary_size = 20000
    pos_vocabulary_size = 100
    label_vocabulary_size = 100
    step_num = 20
    keep_prob = 0.5
    learning_rate = 1.0
    clip_grad = 1

    def __init__(self):
        pass

    def set_batch_size(self, size):
        self.batch_size = size

    def set_corpus_vocabulary_size(self, size):
        self.corpus_vocabulary_size = size

    def set_pos_vocabulary_size(self, size):
        self.pos_vocabulary_size = size

    def set_label_vocabulary_size(self, size):
        self.label_vocabulary_size = size

    def set_step_num(self, num):
        self.step_num = num

    def set_keep_prob(self, value):
        self.keep_prob = value

    def set_learning_rate(self, value):
        self.learning_rate = value

    def set_hidden_size(self, size):
        self.hidden_size = size

    def set_clip_grad(self, value):
        self.clip_grad = value

    def __copy__(self):
        n = Config()
        n.batch_size = self.batch_size
        n.clip_grad = self.clip_grad

        n.corpus_vocabulary_size = self.corpus_vocabulary_size
        n.label_vocabulary_size = self.label_vocabulary_size
        n.pos_vocabulary_size = self.pos_vocabulary_size

        n.hidden_size = self.hidden_size
        n.keep_prob = self.keep_prob
        n.step_num = self.step_num
        n.learning_rate = self.learning_rate
        return n

