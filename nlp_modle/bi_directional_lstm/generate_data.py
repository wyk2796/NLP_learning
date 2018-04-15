# coding:utf-8
import numpy as np


class GenerateData(object):
    """
    generating training_data, test_data for training processes.
    this model need three data set: corpus, pos and label.
    corpus and pos are inputting elements for this model
    """

    def __init__(self, corpus, pos, label):
        assert len(corpus) == len(pos)
        self.corpus = corpus
        self.pos = pos
        self.label = label
        self.max_sentence_length = len(max(corpus, key=lambda x: len(x)))
        self.corpus_size = len(corpus)

    def generate_train_data(self, batch_size):
        step_num = self.max_sentence_length
        batch_num = self.corpus_size // batch_size
        i_corpus = np.zeros((batch_num * batch_size, step_num), dtype=np.int32)
        i_pos = np.zeros((batch_num * batch_size, step_num), dtype=np.int32)
        i_label = np.zeros((batch_num * batch_size, step_num), dtype=np.int32)
        i_sequence_length = np.zeros((batch_num * batch_size))
        for i in range(batch_num * batch_size):
            t_corpus = np.zeros(step_num, dtype=np.int8)
            t_pos = np.zeros(step_num, dtype=np.int8)
            t_label = np.zeros(step_num, dtype=np.int8)
            line_c = self.corpus[i]
            seq_length = len(line_c)
            line_p = self.pos[i]
            line_l = self.label[i]
            for j in range(seq_length):
                t_corpus[j] = line_c[j]
                t_pos[j] = line_p[j]
                t_label[j] = line_l[j]
            i_corpus[i] = t_corpus
            i_pos[i] = t_pos
            i_label[i] = t_label
            i_sequence_length[i] = seq_length
        for i in range(batch_num):
            c_data = i_corpus[batch_size * i: batch_size * (i+1)]
            p_data = i_pos[batch_size * i: batch_size * (i+1)]
            l_data = i_label[batch_size * i: batch_size * (i+1)]
            length = i_sequence_length[batch_size * i: batch_size * (i+1)]
            yield c_data, p_data, l_data, length

    def generate_data_with_bucket(self, buckets_size):
        c_buckets = [list() for _ in range(len(buckets_size) + 1)]
        p_buckets = [list() for _ in range(len(buckets_size) + 1)]
        l_buckets = [list() for _ in range(len(buckets_size) + 1)]
        for i in range(self.corpus_size):
            sen_len = len(self.corpus[i])
            j = 0
            while j < len(buckets_size):
                if sen_len < buckets_size[j]:
                    c_buckets[j].append(self.corpus[i])
                    p_buckets[j].append(self.pos[i])
                    l_buckets[j].append(self.label[i])
                    break
                j += 1
            if j == len(buckets_size):
                c_buckets[j].append(self.corpus[i])
                p_buckets[j].append(self.pos[i])
                l_buckets[j].append(self.label[i])
        return BucketsGenerateData(c_buckets, p_buckets, l_buckets)

    def generate_predict_data(self):
        i_corpus = np.zeros((len(self.corpus), self.max_sentence_length), dtype=np.int32)
        i_pos = np.zeros((len(self.pos), self.max_sentence_length), dtype=np.int32)
        i_sequence_length = np.zeros((len(self.corpus)))
        for i in range(len(self.corpus)):
            t_corpus = np.zeros(self.max_sentence_length, dtype=np.int32)
            t_pos = np.zeros(self.max_sentence_length, dtype=np.int32)
            line_c = self.corpus[i]
            seq_length = len(line_c)
            line_p = self.pos[i]
            for j in range(seq_length):
                t_corpus[j] = line_c[j]
                t_pos[j] = line_p[j]
            i_corpus[i] = t_corpus
            i_pos[i] = t_pos
            i_sequence_length[i] = seq_length
        for i in range(len(self.corpus)):
            c_data = i_corpus[i]
            p_data = i_pos[i]
            length = i_sequence_length[i]
            yield c_data, p_data, length


class BucketsGenerateData(object):

    def __init__(self, c_buckets, p_buckets, l_buckets):
        self.c_buckets = c_buckets
        self.p_buckets = p_buckets
        self.l_buckets = l_buckets
        self.buckets_num = len(c_buckets)
        self.buckets_size = [len(elem) for elem in c_buckets]

    def generate_train_data(self, batch_size):
        select_bucket = np.random.randint(self.buckets_num)
        select_sen = np.random.randint(self.buckets_size[select_bucket], size=batch_size)
        return self.c_buckets[select_sen], self.p_buckets[select_sen], self.l_buckets[select_sen]