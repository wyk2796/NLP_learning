# coding:utf-8
import tensorflow as tf
import numpy as np
import params
from nlp_modle.bi_directional_lstm.generate_data import GenerateData
from nlp_modle.bi_directional_lstm.bidirectional_model import BidirectionalModel
from nlp_modle.bi_directional_lstm.bidirectional_crf import BidirectionalCRF
from nlp_modle.bi_directional_lstm.config import Config
from evaluative_metrics.confusion_metrics import ConfusionMetrics
from evaluative_metrics.record_data import CollecterGroup

def train_model(train_data, train_dicts, test_data, test_dicts, save_path, re_train=False):
    graph = tf.Graph()
    corpus_dict = train_dicts['corpus']
    pos_dict = train_dicts['pos']
    label_dict = train_dicts['label']
    conf = Config()
    conf.set_batch_size(20)
    conf.set_hidden_size(50)
    conf.set_keep_prob(0.5)
    conf.set_learning_rate(1e-3)
    conf.set_corpus_vocabulary_size(corpus_dict.vocabulary_size)
    conf.set_step_num(train_data.max_sentence_length)
    conf.set_label_vocabulary_size(label_dict.vocabulary_size)
    conf.set_pos_vocabulary_size(pos_dict.vocabulary_size)
    epoch = 50

    with graph.as_default():
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        t_model = BidirectionalModel(conf)
        t_model.get_train_graph(initializer)

        test_model = BidirectionalModel(conf)
        test_model.get_test_graph()
    sv = tf.train.Supervisor(graph=graph, logdir=save_path)
    with sv.managed_session() as session:
        if re_train:
            checkpoint = tf.train.get_checkpoint_state(save_path)
            sv.saver.restore(session, checkpoint.model_checkpoint_path)

        for i in range(epoch):
            count = 0
            accuracy = 0
            precision = 0
            for step, (c, p, l, length) in enumerate(train_data.generate_train_data(conf.batch_size)):
                (acc, pre) = t_model.train_run(session, c, p, l, length)
                accuracy += acc
                precision += pre
                count += 1
            print('epoch %d, cost %f, precision %f' % (i, accuracy / count, precision / count))

            if i % 3 == 0 and i != 0:
                test_count = 0
                test_accuracy = 0
                test_precision = 0
                for step, (c, p, l, length) in enumerate(test_data.generate_train_data(conf.batch_size)):
                    (acc, pre) = t_model.test_run(session, c, p, l, length)
                    test_accuracy += acc
                    test_precision += pre
                    test_count += 1
                print('test epoch %d, cost %f, precision %f' % (i, test_accuracy / test_count, test_precision / test_count))


def train_crf_model(train_data, test_data, dicts, seq, save_path, re_train=False):
    graph = tf.Graph()
    corpus_dict = dicts['corpus']
    pos_dict = dicts['pos']
    label_dict = dicts['label']
    conf = Config()
    conf.set_batch_size(40)
    conf.set_hidden_size(200)
    conf.set_keep_prob(0.5)
    conf.set_learning_rate(1e-3)
    conf.set_corpus_vocabulary_size(corpus_dict.vocabulary_size)
    conf.set_step_num(train_data.max_sentence_length)
    conf.set_label_vocabulary_size(label_dict.vocabulary_size)
    conf.set_pos_vocabulary_size(pos_dict.vocabulary_size)
    epoch = 150

    cor = corpus_dict.text_transition_word2id(seq[0])
    po = pos_dict.text_transition_word2id(seq[1])
    pd = GenerateData(cor, po, None)
    p_conf = conf.__copy__()
    p_conf.set_batch_size(1)
    p_conf.set_step_num(pd.max_sentence_length)

    with graph.as_default():
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        t_model = BidirectionalCRF(conf)
        t_model.get_crf_train_without_pos_graph(initializer)

        test_model = BidirectionalCRF(conf)
        test_model.get_crf_test_without_pos_graph()

        # p_model = BidirectionalCRF(p_conf)
        # p_model.get_crf_predict_graph()
    sv = tf.train.Supervisor(graph=graph, logdir=save_path)
    with sv.managed_session() as session:
        if re_train:
            checkpoint = tf.train.get_checkpoint_state(save_path)
            sv.saver.restore(session, checkpoint.model_checkpoint_path)

        collecter = CollecterGroup('BiLSTM_CRF')
        collecter.add_collecter('Train Average')
        collecter.add_collecter('Test Average')

        for i in range(epoch):
            count = 0
            loss = 0
            cm = ConfusionMetrics(range(conf.label_vocabulary_size))
            for step, (c, p, l, length) in enumerate(train_data.generate_train_data(conf.batch_size)):
                (ll, pre) = t_model.train_run(session, c, p, l, length)
                loss += ll
                cm.computer_confusion_matrix(np.reshape(pre, [-1]), np.reshape(l, [-1]))
                count += 1
            precision = cm.class_at_id_precision()[4:]
            recall = cm.class_at_id_recall()[4:]
            f1 = cm.class_f1_measure()[4:]
            s_p = sum(precision) / 8
            s_r = sum(recall) / 8
            s_f1 = 2 * s_p * s_r / (s_p + s_r)
            for step, (p, r, f) in enumerate(zip(precision, recall, f1)):
                print('label %d, precision %f, recall %f; F1 %f ' % (step, p, r, f))
            print('epoch %d, cost %f, mp: %f, mc: %f, f1: %f ' %
                  (i, loss / count, s_p, s_r,  s_f1))
            collecter.add_value_by_name('Train Average', s_p, s_r, s_f1, loss / count)

            if i % 3 == 0 and i != 0:
                test_count = 0
                test_loss = 0
                test_c = ConfusionMetrics(range(conf.label_vocabulary_size))
                for step, (c, p, l, length) in enumerate(test_data.generate_train_data(conf.batch_size)):
                    (ll, pre) = test_model.test_run(session, c, p, l, length)
                    test_loss += ll
                    test_c.computer_confusion_matrix(np.reshape(pre, [-1]), np.reshape(l, [-1]))
                    test_count += 1
                test_precision = test_c.class_at_id_precision()[4:]
                test_recall = test_c.class_at_id_recall()[4:]
                test_f1 = test_c.class_f1_measure()[4:]
                test_s_p = sum(test_precision) / 8
                test_s_r = sum(test_recall) / 8
                test_s_f1 = 2 * test_s_p * test_s_r / (test_s_p + test_s_r)
                msg_test = ''
                for step, (p, r, f) in enumerate(zip(test_precision, test_recall, test_f1)):
                    msg_test += ('label %d, precision %f, recall %f; F1 %f' % (step, p, r, f))
                print('test epoch %d, cost %f, mp: %f, mc: %f, f1: %f detail:%s ' %
                      (i, test_loss / test_count, test_s_p, test_s_r, test_s_f1, msg_test))
                collecter.add_value_by_name('Test Average', test_s_p, test_s_r, test_s_f1, test_loss / test_count)
        collecter.putout_figure(params.conll_figure_path)
        # for step, (c, p, length) in enumerate(pd.generate_predict_data()):
        #     seq = p_model.predict_run(session, c, p, length)
        #     corpus = corpus_dict.text_transition_id2word([c])
        #     lab = label_dict.text_transition_id2word(seq)
        #     print(step)
        #     print(corpus)
        #     print(lab)

def predict_crf_model(data, dicts, save_path):
    graph = tf.Graph()
    corpus_dict = dicts['corpus']
    pos_dict = dicts['pos']
    label_dict = dicts['label']
    conf = Config()
    conf.set_batch_size(1)
    conf.set_hidden_size(200)
    conf.set_keep_prob(0.5)
    conf.set_learning_rate(1e-3)
    conf.set_batch_size(1)
    conf.set_step_num(data.max_sentence_length)
    conf.set_corpus_vocabulary_size(corpus_dict.vocabulary_size)
    conf.set_pos_vocabulary_size(pos_dict.vocabulary_size)

    with graph.as_default():
        # initializer = tf.random_uniform_initializer(-0.1, 0.1)
        p_model = BidirectionalCRF(conf)
        p_model.get_crf_predict_graph()

    sv = tf.train.Supervisor(graph=graph, logdir=save_path)
    with sv.managed_session() as session:
        checkpoint = tf.train.get_checkpoint_state(save_path)
        sv.saver.restore(session, checkpoint.model_checkpoint_path)

        for step, (c, p, l, length) in enumerate(data.generate_predict_data()):
            p = data.test_run(session, c, p, l, length)
            label_dict.text_transition_id2word(p)

        print('seq %s' % ' '.join(label_dict))

