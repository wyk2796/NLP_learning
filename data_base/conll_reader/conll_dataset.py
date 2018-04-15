# coding:utf-8
from data_base.data_load import Read
from data_base import util

class Conll(Read):

    corpus = []
    pos = []
    chunk_tags = []
    entity_name_tags = []
    line_num = 0

    _temp_corpus = []
    _temp_pos = []
    _temp_chunk_tags = []
    _temp_entity_name_tags = []

    filter_corpus = ['-DOCSTART-']

    def _read_line(self, line):
        line = line.strip()
        if line == '':
            if len(self._temp_corpus) > 0:
                self.corpus.append(self._temp_corpus.copy())
                self.pos.append(self._temp_pos.copy())
                self.chunk_tags.append(self._temp_chunk_tags.copy())
                self.entity_name_tags.append(self._temp_entity_name_tags.copy())
                self._temp_corpus.clear()
                self._temp_pos.clear()
                self._temp_chunk_tags.clear()
                self._temp_entity_name_tags.clear()
                self.line_num += 1
        else:
            arr_str = line.split(' ')
            if arr_str[0] not in self.filter_corpus:
                self._temp_corpus.append(arr_str[0])
                self._temp_pos.append(arr_str[1])
                self._temp_chunk_tags.append(arr_str[2])
                self._temp_entity_name_tags.append((arr_str[3]))

    def get_corpus(self):
        return self.corpus

    def get_pos(self):
        return self.pos

    def get_chunk_tags(self):
        return self.chunk_tags

    def get_name_entity_tags(self):
        return self.entity_name_tags

    def save_data(self, save_dir):
        corpus_file = save_dir + '/corpus.txt'
        pos_file = save_dir + '/pos.txt'
        label_file = save_dir + '/label.txt'
        util.save_to_file(corpus_file, self.corpus)
        util.save_to_file(pos_file, self.pos)
        util.save_to_file(label_file, self.entity_name_tags)


class ConllExtend(Read):

    corpus = {}
    _temp_corpus = {}
    line_num = 0
    filter_corpus = ['-DOCSTART-']

    def _read_line(self, line):
        line = line.strip()
        if line == '':
            if len(self._temp_corpus) > 0:
                for i in range(len(self._temp_corpus)):
                    pre = self.corpus.get(i, list())
                    pre.append(self._temp_corpus[i].copy())
                    self.corpus[i] = pre
                self._temp_corpus.clear()
                self.line_num += 1
        else:
            arr_str = line.split(' ')
            if arr_str[0] not in self.filter_corpus:
                for i in range(len(arr_str)):
                    pre = self._temp_corpus.get(i, list())
                    pre.append(arr_str[i])
                    self._temp_corpus[i] = pre

    def _end_read(self):
        if len(self._temp_corpus) > 0:
            for i in range(len(self._temp_corpus)):
                pre = self.corpus.get(i, list())
                pre.append(self._temp_corpus[i].copy())
                self.corpus[i] = pre
            self._temp_corpus.clear()
            self.line_num += 1

    def get_corpus(self):
        return self.corpus
