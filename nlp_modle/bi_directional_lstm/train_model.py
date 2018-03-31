# coding:utf-8
import tensorflow as tf
from nlp_modle.bi_directional_lstm.bidirectional_model import BidirectionalModel
from nlp_modle.bi_directional_lstm.bidirectional_crf import BidirectionalCRF
from nlp_modle.bi_directional_lstm.config import Config


def train_model(train_data, train_dicts, test_data, test_dicts, save_path, re_train=False):
    graph = tf.Graph()
    corpus_dict = train_dicts['corpus']
    pos_dict = train_dicts['pos']
    label_dict = train_dicts['label']
    conf = Config()
    conf.set_batch_size(20)
    conf.set_hidden_size(200)
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


def train_crf_model(train_data, test_data, dicts, save_path, re_train=False):
    graph = tf.Graph()
    corpus_dict = dicts['corpus']
    pos_dict = dicts['pos']
    label_dict = dicts['label']
    conf = Config()
    conf.set_batch_size(20)
    conf.set_hidden_size(200)
    conf.set_keep_prob(0.5)
    conf.set_learning_rate(1e-3)
    conf.set_corpus_vocabulary_size(corpus_dict.vocabulary_size)
    conf.set_step_num(train_data.max_sentence_length)
    conf.set_label_vocabulary_size(label_dict.vocabulary_size)
    conf.set_pos_vocabulary_size(pos_dict.vocabulary_size)
    epoch = 50

    with graph.as_default():
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        t_model = BidirectionalCRF(conf)
        t_model.get_crf_train_without_pos_graph(initializer)

        test_model = BidirectionalCRF(conf)
        test_model.get_crf_test_without_pos_graph()
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
                    (acc, pre) = test_model.test_run(session, c, p, l, length)
                    test_accuracy += acc
                    test_precision += pre
                    test_count += 1
                print('test epoch %d, cost %f, precision %f' % (i, test_accuracy / test_count, test_precision / test_count))

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
    conf.set_corpus_vocabulary_size(corpus_dict.vocabulary_size)
    conf.set_step_num(data.max_sentence_length)
    conf.set_label_vocabulary_size(label_dict.vocabulary_size)
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

