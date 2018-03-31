# coding:utf-8
import collections
import data_base.util


class Vocabulary(object):

    def __init__(self, vocabulary_size=10000, default_sign=True):
        self.word2id = {}
        self.id2word = {}
        self.vocabulary_size = vocabulary_size
        self.default_sign = default_sign
        self.start_sign = "_START"
        self.end_sign = "_END"
        self.unknown = "_UNKNOWN"
        self.number_char = "_NUM"

    def create_vocabulary(self, content):
        """
        generate the vocabulary index from original text
        :param content: original text, such as [[I love you !], [yes, me too!]]
        :param default_sign: if default_sign equal true, it means that the vocabulary contain the
               default_sign, such as "_END", "_UNKNOWN", "_NUM". if not, the vocabulary just include
               the word in inputting text.
        :return: none
        """
        word_map = data_base.util.word_frp_statistic(content)
        if self.default_sign:
            count = [[self.start_sign, 1],
                     [self.end_sign, -1],
                     [self.unknown, -1],
                     [self.number_char, -1]]
            count.extend(collections.Counter(word_map).most_common(self.vocabulary_size - 4))
            for w, _ in count:
                if not data_base.util.is_num(w):
                    self.word2id[w] = len(self.word2id)
            self.id2word = dict(zip(self.word2id.values(), self.word2id.keys()))
            self.vocabulary_size = min([len(self.word2id), self.vocabulary_size])
        else:
            count = []
            count.extend(collections.Counter(word_map).most_common(self.vocabulary_size))
            for w, _ in count:
                self.word2id[w] = len(self.word2id)
            self.id2word = dict(zip(self.word2id.values(), self.word2id.keys()))
            self.vocabulary_size = min([len(self.word2id), self.vocabulary_size])

    def text_transition_word2id(self, content):
        """
        transform tokens to ids, such as [[I love you!],[Hello!]] to [[12, 30, 40, 9],[100, 56]]
        :param content: tokens list
        :return: ids list
        """

        assert(len(self.word2id) > 0)
        data = []
        for line in content:
            line_id = []
            for w in line:
                if self.default_sign:
                    line_id.append(self._char2id_transition(w))
                else:
                    line_id.append(self.word2id[w])
            data.append(line_id)
        return data

    def text_transition_id2word(self, content):
        """
        transform tokens to ids, such as [[12, 30, 40, 9],[100, 56]] to [[I love you!],[Hello!]]
        :param content: ids list
        :return: tokens list
        """

        assert(len(self.id2word) > 0)
        data = []
        for line in content:
            line_word = []
            for _id in line:
                line_word.append(self.id2word[_id])
            data.append(line_word)
        return data

    def _char2id_transition(self, c):
        if data_base.util.is_num(c):
            return self.word2id[self.number_char]
        elif c in self.word2id:
            return self.word2id[c]
        else:
            return self.word2id[self.unknown]

    def save_vocabulary_to_file(self, path):
        with open(path, encoding='utf-8', mode='w') as p:
            [p.write(w + ' ' + str(index) + '\n')
             for w, index in sorted(self.word2id.items(), key=lambda x:x[1])]

    def load_vocabulary_from_file(self, path):
        with open(path, encoding='utf-8', mode='r') as r:
            for line in r.readlines():
                line = line.strip()
                if line != '':
                    try:
                        w_pair = line.split(' ')
                        self.word2id[w_pair[0]] = int(w_pair[1])
                    except Exception as e:
                        print('Read line %s, Encounter Error %s' % (line, e.args))
        self.id2word = dict(zip(self.word2id.values(), self.word2id.keys()))











