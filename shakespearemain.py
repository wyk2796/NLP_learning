# coding:utf-8
from data_base.data_load import ShakespeareRead
from data_base import util, VocabularyList
import params

if __name__ == '__main__':
    shakes_reader = ShakespeareRead(params.ShakespeareText, encoding='utf-16-le')
    text = shakes_reader.get_content()
    data = util.transform_lines_list(text)
    dict_list = VocabularyList.Vocabulary(vocabulary_size=20000)
    dict_list.create_vocabulary(data)
    ids_text = dict_list.text_transition_word2id(data)
    [print(str(line)) for line in ids_text]
    words_text = dict_list.text_transition_id2word(ids_text)
    [print(str(line)) for line in words_text]
    dict_list.save_vocabulary_to_file(params.ShakespeareDir + '\\' + 'shake.txt')

