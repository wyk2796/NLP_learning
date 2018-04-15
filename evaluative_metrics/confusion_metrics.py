# coding:utf-8
import numpy as np


class ConfusionMetrics(object):

    def __init__(self, class_id=list()):
        self.class_id = class_id
        self.tagger_num = len(class_id)
        self.matrix = np.zeros(shape=[self.tagger_num, self.tagger_num])
        self.tagger_map = {}
        self.initial_map()

    def initial_map(self):
        for step, tagger in enumerate(self.class_id):
            self.tagger_map[tagger] = step

    def computer_confusion_matrix(self, y_real, y_predict):
        assert len(y_real) == len(y_predict)
        for y_r, y_p in zip(y_real, y_predict):
            self.__counter(y_r, y_p)

    def __counter(self, y_r, y_p):
        x, y = self.tagger_map[y_r], self.tagger_map[y_p]
        self.matrix[x][y] += 1

    def __tagger2ids(self, tagger):
        if tagger is None:
            return None
        return [self.tagger_map[t] for t in tagger]

    def __class_precision(self, ids=None):
        if ids is None:
            return np.diag(self.matrix) / np.clip(np.sum(self.matrix, axis=1), a_min=1, a_max=None)
        else:
            return np.diag(self.matrix)[ids] / np.clip(np.sum(self.matrix[ids, :], axis=1), a_min=1, a_max=None)

    def class_at_id_precision(self, tagger=None):
        return self.__class_precision(self.__tagger2ids(tagger))

    def __class_recall(self, ids=None):
        if ids is None:
            return np.diag(self.matrix) / np.clip(np.sum(self.matrix, axis=0), a_min=1, a_max=None)
        else:
            return np.diag(self.matrix)[ids] / np.clip(np.sum(self.matrix[:, ids], axis=0), a_min=1, a_max=None)

    def class_at_id_recall(self, tagger=None):
        return self.__class_recall(self.__tagger2ids(tagger))

    def class_f1_measure(self, tagger=None):
        ids = self.__tagger2ids(tagger)
        p = self.__class_precision(ids)
        r = self.__class_recall(ids)
        return 2 * p * r / np.clip(p + r, a_min=1e-6, a_max=None)
