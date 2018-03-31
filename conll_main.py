# coding:utf-8
from data_base.conll_reader.conll_dataset import Conll
import params
from data_base.VocabularyList import Vocabulary
from nlp_modle.bi_directional_lstm.generate_data import GenerateData
from nlp_modle.bi_directional_lstm import train_model


def create_dict_with_save(data, c_dict_path, p_dict_path, l_dict_path):
    corpus_dict = Vocabulary(vocabulary_size=30000)
    pos_dict = Vocabulary()
    label_dict = Vocabulary()
    corpus_dict.create_vocabulary(data.corpus)
    corpus_dict.save_vocabulary_to_file(c_dict_path)
    pos_dict.create_vocabulary(data.pos)
    pos_dict.save_vocabulary_to_file(p_dict_path)
    label_dict.create_vocabulary(data.entity_name_tags)
    label_dict.save_vocabulary_to_file(l_dict_path)


def load_dict_from_file(dict_path, dict):
    return dict.load_vocabulary_from_file(dict_path)


def prepare_data(path, c_dict_path, p_dict_path, l_dict_path):
    data = Conll(path)
    corpus_dict = Vocabulary(vocabulary_size=30000)
    pos_dict = Vocabulary()
    label_dict = Vocabulary()

    corpus_dict.load_vocabulary_from_file(c_dict_path)
    pos_dict.load_vocabulary_from_file(p_dict_path)
    label_dict.load_vocabulary_from_file(l_dict_path)

    ids_c = corpus_dict.text_transition_word2id(data.corpus)
    ids_p = pos_dict.text_transition_word2id(data.pos)
    ids_l = label_dict.text_transition_word2id(data.entity_name_tags)

    dicts = {'corpus': corpus_dict, 'pos': pos_dict, 'label': label_dict}
    td = GenerateData(ids_c, ids_p, ids_l)
    # td.generate_data_with_bucket([5, 10, 20, 30, 40, 50, 55])
    return td, dicts

if __name__ == '__main__':
    (train_date, train_dict) = prepare_data(
        params.conll_train, params.conll_corpus_dict, params.conll_pos_dict, params.conll_label_dict)
    (test_data, test_dict) = prepare_data(
        params.conll_testa, params.conll_corpus_dict, params.conll_pos_dict, params.conll_label_dict, False)
    # train_model.train_model(train_date, train_dict, test_data, test_dict, params.conll_model_path)
    train_model.train_crf_model(train_date, test_data, train_dict, test_dict, params.conll_model_path)
