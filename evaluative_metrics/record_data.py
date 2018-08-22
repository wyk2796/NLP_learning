# coding:utf-8
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import params
import os


class Record:

    def __init__(self, name):
        self.name = name
        self.__record = []

    def add(self, value):
        self.__record.append(value)

    def get_value(self):
        return self.__record

    def length(self):
        return len(self.__record)


class CollectMetrics:

    def __init__(self, name):
        self.name = name
        self._precision = Record('precision')
        self._recall = Record('recall')
        self._f1 = Record('f1')
        self._loss = Record('loss')

    def append(self, precision, recall, f1, loss):
        self._precision.add(precision) if precision is not None else 0
        self._recall.add(recall) if recall is not None else 0
        self._f1.add(f1) if recall is not None else 0
        self._loss.add(loss) if recall is not None else 0

    def draw_all(self, save_path=None):
        plt.figure(self.name)
        plt.title('BiLSTM-CRF ' + self.name + ' precision recall f1')
        plt.xlabel('epoch')
        plt.ylabel('rate')
        plt.ylim(0, 1)

        if self._precision.length() is not 0:
            plt.plot(self._precision.get_value(),
                     color='r',
                     linewidth=2.5,
                     linestyle='-',
                     label='precision')
        if self._recall.length() is not 0:
            plt.plot(self._recall.get_value(),
                     color='b',
                     linewidth=2.5,
                     linestyle='-',
                     label='recall')
        if self._f1.length() is not 0:
            plt.plot(self._f1.get_value(),
                     color='g',
                     linewidth=2.5,
                     linestyle='-',
                     label='f1')
        plt.legend(loc='lower rightz')
        if save_path is None:
            plt.show()
        else:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            plt.savefig(save_path + '/' + self.name + '_metrics.png')
        self.draw_loss(save_path)

    def draw_loss(self, save_path=None):
        if self._loss.length() is not 0:
            plt.figure(self.name + '_loss')
            plt.title('BiLSTM-CRF ' + self.name + ' Loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.ylim(0, 10)
            plt.plot(self._loss.get_value(),
                     color='b',
                     linewidth=2.5,
                     linestyle='-',
                     label='loss')
            plt.legend(loc='upper left')
            if save_path is None:
                plt.show()
            else:
                plt.savefig(save_path + '/' + self.name + '_loss.png')


class CollecterGroup:

    def __init__(self, name):
        self.name = name
        self.collecter = {}

    def add_collecter(self, col_name):
        self.collecter[col_name] = CollectMetrics(col_name)

    def add_value_by_name(self, col_name, precision, recall, f1, loss=None):
        if col_name in self.collecter:
            self.collecter[col_name].append(precision, recall, f1, loss)

    def putout_figure(self, save_path):
        for (col_name, c) in self.collecter.items():
            c.draw_all(save_path + self.name)


if __name__ == '__main__':
    def getvalue():
        prf = np.random.rand(3)
        loss = np.random.rand() * 10
        return prf, loss
    c = CollecterGroup('seq2seq')
    c.add_collecter('label1')
    c.add_collecter('label2')
    c.add_collecter('label3')
    for i in range(100):
        prf, loss = getvalue()
        c.add_value_by_name('label1', prf[0], prf[1], prf[2], loss)
        prf, loss = getvalue()
        c.add_value_by_name('label2', prf[0], prf[1], prf[2])
        prf, loss = getvalue()
        c.add_value_by_name('label3', prf[0], prf[1], prf[2])
    c.putout_figure(params.conll_figure_path)