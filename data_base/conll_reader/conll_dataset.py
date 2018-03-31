# coding:utf-8
from data_base.data_load import Read


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





