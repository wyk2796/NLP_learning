# coding:utf-8
import tensorflow as tf
import tensorflow.contrib.crf as crf
from nlp_modle.bi_directional_lstm import bi_attention


class BidirectionalCRF(object):
    def __init__(self, config):
        self.config = config
        self.hidden_size = config.hidden_size
        self.learning_rate = config.learning_rate
        self.clip_grad = config.clip_grad

        self.corpus_vocabulary_size = config.corpus_vocabulary_size
        self.pos_vocabulary_size = config.pos_vocabulary_size
        self.label_vocabulary_size = config.label_vocabulary_size

        self.step_num = config.step_num
        self.batch_size = config.batch_size

        self.embed_corpus = None
        self.embed_pos = None

        self.status = None
        self.keep_prob = config.keep_prob

        self.corpus_input = None
        self.pos_input = None
        self.label_input = None
        self.sequence_length = None

        self.transition_params = None

        self.logits = None
        self.cost = None
        self.train_op = None
        self.predict = None
        # self.merged = None
        # self.precision = None

    def _lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(
            self.hidden_size, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)

    def add_placeholder(self):
        self.corpus_input = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.step_num])
        self.pos_input = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.step_num])
        self.label_input = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.step_num])
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])

    def forwards(self):
        with tf.variable_scope('embedding'):
            self.embed_corpus = tf.get_variable('embed_corpus',
                                                shape=[self.corpus_vocabulary_size, self.hidden_size])
            self.embed_pos = tf.get_variable('embed_pos',
                                             shape=[self.pos_vocabulary_size, self.hidden_size])

        with tf.device("/cpu:0"):
            input_data = tf.nn.embedding_lookup(self.embed_corpus, self.corpus_input)
            pos_data = tf.nn.embedding_lookup(self.embed_pos, self.pos_input)
        if self.status == 'train' and self.keep_prob < 1.0:
            input_data = tf.nn.dropout(input_data, keep_prob=self.keep_prob)
            pos_data = tf.nn.dropout(pos_data, keep_prob=self.keep_prob)

        """
        encoding corpus:
        """

        (corpus_fw, corpus_bw), _ = tf.nn.bidirectional_dynamic_rnn(self._lstm_cell(),
                                                                    self._lstm_cell(),
                                                                    input_data,
                                                                    dtype=tf.float32,
                                                                    # time_major=True,
                                                                    sequence_length=self.sequence_length,
                                                                    scope='encode_corpus')
        con_corpus = tf.concat([corpus_fw, corpus_bw], axis=-1)
        """
        encode pos
        """

        (pos_fw, pos_bw), _ = tf.nn.bidirectional_dynamic_rnn(self._lstm_cell(),
                                                              self._lstm_cell(),
                                                              pos_data,
                                                              dtype=tf.float32,
                                                              # time_major=True,
                                                              sequence_length=self.sequence_length,
                                                              scope='encode_pos')
        con_pos = tf.concat([pos_fw, pos_bw], axis=-1)
        result = tf.concat([con_corpus, con_pos], axis=-1)

        if self.status == 'train':
            result = tf.nn.dropout(result, keep_prob=self.keep_prob)

        with tf.variable_scope('output_layer_crf', reuse=None):
            u_w = tf.get_variable('U_weight', shape=[self.hidden_size * 4, self.label_vocabulary_size])
            u_b = tf.get_variable('U_bias', shape=[self.label_vocabulary_size])

        logits_list = [tf.matmul(result[i], u_w) + u_b for i in range(self.batch_size)]
        self.logits = tf.concat(logits_list, axis=0)

    def forwards_seq2seq(self):
        with tf.variable_scope('embedding'):
            self.embed_corpus = tf.get_variable('embed_corpus',
                                                shape=[self.corpus_vocabulary_size, self.hidden_size])
            self.embed_pos = tf.get_variable('embed_pos',
                                             shape=[self.pos_vocabulary_size, self.hidden_size])

        with tf.device("/cpu:0"):
            input_data = tf.nn.embedding_lookup(self.embed_corpus, self.corpus_input)
            pos_data = tf.nn.embedding_lookup(self.embed_pos, self.pos_input)
        if self.status == 'train' and self.keep_prob < 1.0:
            input_data = tf.nn.dropout(input_data, keep_prob=self.keep_prob)
            pos_data = tf.nn.dropout(pos_data, keep_prob=self.keep_prob)

        """
        encoding corpus:
        """

        _, (s_fw, s_bw) = tf.nn.bidirectional_dynamic_rnn(self._lstm_cell(),
                                                          self._lstm_cell(),
                                                          input_data,
                                                          dtype=tf.float32,
                                                          # time_major=True,
                                                          sequence_length=self.sequence_length,
                                                          scope='encode_corpus')
        # con_corpus = tf.concat([corpus_fw, corpus_bw], axis=-1)
        """
        encode pos
        """

        (pos_fw, pos_bw), _ = tf.nn.bidirectional_dynamic_rnn(self._lstm_cell(),
                                                              self._lstm_cell(),
                                                              pos_data,
                                                              dtype=tf.float32,
                                                              initial_state_fw=s_fw,
                                                              initial_state_bw=s_bw,
                                                              sequence_length=self.sequence_length,
                                                              scope='encode_pos')
        result = tf.concat([pos_fw, pos_bw], axis=-1)
        # result = tf.concat([con_corpus, con_pos], axis=-1)

        if self.status == 'train':
            result = tf.nn.dropout(result, keep_prob=self.keep_prob)

        with tf.variable_scope('output_layer_crf', reuse=None):
            u_w = tf.get_variable('U_weight', shape=[self.hidden_size * 2, self.label_vocabulary_size])
            u_b = tf.get_variable('U_bias', shape=[self.label_vocabulary_size])

        logits_list = [tf.matmul(result[i], u_w) + u_b for i in range(self.batch_size)]
        self.logits = tf.concat(logits_list, axis=0)

    def forward_BIRNN_corpus(self):
        with tf.variable_scope('embedding'):
            self.embed_corpus = tf.get_variable('embed_corpus',
                                                shape=[self.corpus_vocabulary_size, self.hidden_size])

        with tf.device("/cpu:0"):
            input_data = tf.nn.embedding_lookup(self.embed_corpus, self.corpus_input)
        if self.status == 'train' and self.keep_prob < 1.0:
            input_data = tf.nn.dropout(input_data, keep_prob=self.keep_prob)

        """
        encoding corpus:
        """
        (corpus_fw, corpus_bw), _ = tf.nn.bidirectional_dynamic_rnn(self._lstm_cell(),
                                                                    self._lstm_cell(),
                                                                    input_data,
                                                                    dtype=tf.float32,
                                                                    # time_major=True,
                                                                    sequence_length=self.sequence_length,
                                                                    scope='encode_corpus')
        con_corpus = tf.concat([corpus_fw, corpus_bw], axis=-1)
        if self.status == 'train':
            con_corpus = tf.nn.dropout(con_corpus, keep_prob=self.keep_prob)

        with tf.variable_scope('output_layer', reuse=None):
            u_w = tf.get_variable('U_weight', shape=[self.hidden_size * 2, self.label_vocabulary_size])
            u_b = tf.get_variable('U_bias', shape=[self.label_vocabulary_size])

        logits_list = [tf.matmul(con_corpus[i], u_w) + u_b for i in range(self.batch_size)]
        self.logits = tf.concat(logits_list, axis=0)

    def forward_attention(self):
        with tf.variable_scope('embedding'):
            self.embed_corpus = tf.get_variable('embed_corpus',
                                                shape=[self.corpus_vocabulary_size, self.hidden_size])
            self.embed_pos = tf.get_variable('embed_pos',
                                             shape=[self.pos_vocabulary_size, self.hidden_size])

        with tf.device("/cpu:0"):
            input_data = tf.nn.embedding_lookup(self.embed_corpus, self.corpus_input)
            pos_data = tf.nn.embedding_lookup(self.embed_pos, self.pos_input)
        if self.status == 'train' and self.keep_prob < 1.0:
            input_data = tf.nn.dropout(input_data, keep_prob=self.keep_prob)
            pos_data = tf.nn.dropout(pos_data, keep_prob=self.keep_prob)

        """
        encoding corpus:
        """

        (corpus_fw, corpus_bw), (c_fs, c_bs) = tf.nn.bidirectional_dynamic_rnn(self._lstm_cell(),
                                                                               self._lstm_cell(),
                                                                               input_data,
                                                                               dtype=tf.float32,
                                                                               # time_major=True,
                                                                               sequence_length=self.sequence_length,
                                                                               scope='encode_corpus')
        con_corpus = tf.concat([corpus_fw, corpus_bw], axis=-1)
        """
        encode pos
        """

        (pos_fw, pos_bw), _ = tf.nn.bidirectional_dynamic_rnn(self._lstm_cell(),
                                                              self._lstm_cell(),
                                                              pos_data,
                                                              dtype=tf.float32,
                                                              initial_state_fw=c_fs,
                                                              initial_state_bw=c_bs,
                                                              # time_major=True,
                                                              sequence_length=self.sequence_length,
                                                              scope='encode_pos')
        con_pos = tf.concat([pos_fw, pos_bw], axis=-1)

        result = bi_attention.attention_layer(con_corpus, con_pos, self.batch_size, self.step_num, self.hidden_size)

        result = tf.reshape(tf.concat(result, axis=0), [self.batch_size, self.step_num, -1])
        # result = tf.concat([con_corpus, con_pos], axis=-1)

        if self.status == 'train':
            result = tf.nn.dropout(result, keep_prob=self.keep_prob)

        with tf.variable_scope('output_layer_crf', reuse=None):
            u_w = tf.get_variable('U_weight', shape=[self.hidden_size * 2, self.label_vocabulary_size])
            u_b = tf.get_variable('U_bias', shape=[self.label_vocabulary_size])

        logits_list = [tf.matmul(result[i], u_w) + u_b for i in range(self.batch_size)]
        self.logits = tf.concat(logits_list, axis=0)

    # def attention_context(self, con_corpus, con_pos):
    #     with tf.variable_scope('attention'):
    #         v_t = tf.get_variable('v_t', shape=[self.hidden_size])
    #         w_a = tf.get_variable('w_a', shape=[self.hidden_size * 4, self.hidden_size])
    #         convect = []
    #         print(con_pos.shape)
    #         for pos in tf.split(tf.transpose(con_pos, [1, 0, 2]), self.step_num):
    #             e = [tf.concat([pos, corpus], axis=2) for corpus in tf.split(tf.transpose(con_corpus, [1, 0, 2]), self.step_num)]
    #             e = tf.reshape(tf.concat(e, axis=0), shape=[self.step_num, self.batch_size, self.hidden_size * 4])
    #             score = [tf.reduce_sum(v_t * tf.nn.tanh(tf.matmul(e[i], w_a)), axis=1) for i in range(self.step_num)]
    #             score = tf.reshape(tf.concat(score, axis=0), [self.batch_size, self.step_num])
    #             score = tf.nn.softmax(score)
    #             convect.append(score)
    #         a = tf.reshape(tf.concat(convect, axis=0), [self.batch_size, self.step_num, self.step_num])
    #         print(a.shape)
    #         h_j = tf.reshape(tf.concat(con_corpus, axis=0), [self.batch_size, self.step_num, 2 * self.hidden_size])
    #         h_j_e = tf.expand_dims(h_j, axis=1)
    #         print(h_j_e.shape)
    #         h_j = tf.concat([h_j_e] * self.step_num, axis=1)
    #         c = h_j * a
    #         print(c.shape)
    #         raise Exception('')
    #         context = []
    #         for i in range(self.batch_size):
    #             batch_h_j = h_j[i]
    #             batch_a = a[i]
    #             seq_c = []
    #             for j in range(self.step_num):
    #                 seq_a = batch_a[j]
    #                 c_t = [seq_a[k] * batch_h_j[k] for k in range(self.step_num)]
    #                 seq_c.append(c_t)
    #             tf.reduce_sum(tf.concat(seq_c, axis=0), axis=0)
    #             context.append(seq_c)
    #         return context

    def computer_crf_loss(self):
        with tf.variable_scope('crf', reuse=None):
            likelihood, self.transition_params = crf.crf_log_likelihood(
                tf.reshape(self.logits, [self.batch_size, -1, self.label_vocabulary_size]),
                self.label_input,
                self.sequence_length)
            self.cost = tf.reduce_mean(-likelihood)

    def computer_bidirectional_loss(self):
        label = tf.reshape(tf.one_hot(self.label_input, self.label_vocabulary_size), [-1, self.label_vocabulary_size])
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                              labels=label))
        self.cost = self.cost / self.batch_size

    def computer_train_Adam(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate)
        self._clip_grad()

    def computer_train_SGD(self):
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate)
        self._clip_grad()

    def _clip_grad(self):
        if self.train_op is not None and self.cost is not None:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            grad_and_vars = self.train_op.compute_gradients(self.cost)
            self.train_op = self.train_op.apply_gradients([[
                tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v]
                for g, v in grad_and_vars
            ], global_step=self.global_step)

    def computer_bidirectional_predict(self):
        self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict_output')
        self.predict = tf.reshape(self.predict, shape=[self.batch_size, self.step_num])

    def computer_crf_predict(self):
        self.predict, _ = crf.crf_decode(
            tf.reshape(self.logits, [self.batch_size, -1, self.label_vocabulary_size]),
            self.transition_params,
            self.sequence_length)

    # def computer_precision_rate(self):
    #     """
    #     output every tagger's precision
    #     :return:
    #     """
    #     _, self.precision = tf.metrics.mean_per_class_accuracy(
    #         tf.cast(self.label_input, tf.int64),
    #         self.predict, num_classes=self.label_vocabulary_size)
        # batch_seq_mask = tf.sequence_mask(self.sequence_length, maxlen=self.step_num)
        # correct_predict = tf.equal(tf.cast(self.predict, tf.int32), self.label_input)
        # reduce_value = tf.boolean_mask(correct_predict, batch_seq_mask)
        # self.precision = tf.reduce_mean(tf.cast(reduce_value, tf.float32))

    # def summary_operation(self):
    #     tf.summary.scalar('loss', self.cost)
    #     tf.summary.histogram('loss_histogram', self.cost)
    #     self.merged = tf.summary.merge_all()
    #     print(self.merged)

    def get_crf_train_graph(self, init_variable):
        with tf.name_scope('Train'):
            with tf.variable_scope('model', reuse=None, initializer=init_variable):
                self.status = 'train'
                self.add_placeholder()
                self.forwards()
                self.computer_crf_loss()
                self.computer_train_Adam()
                self.computer_crf_predict()

    def get_crf_test_graph(self):
        with tf.name_scope('Test'):
            with tf.variable_scope('model', reuse=True):
                self.status = 'test'
                self.add_placeholder()
                self.forwards()
                self.computer_crf_loss()
                self.computer_crf_predict()

    def get_bidirectional_train_graph(self, init_variable):
        with tf.name_scope('Train'):
            with tf.variable_scope('model', reuse=None, initializer=init_variable):
                self.status = 'train'
                self.add_placeholder()
                self.forwards()
                self.computer_bidirectional_loss()
                self.computer_train_Adam()
                self.computer_bidirectional_predict()

    def get_bidirectional_test_graph(self):
        with tf.name_scope('Test'):
            with tf.variable_scope('model', reuse=True):
                self.status = 'test'
                self.add_placeholder()
                self.forwards()
                self.computer_bidirectional_loss()
                self.computer_bidirectional_predict()

    def get_crf_train_without_pos_graph(self, init_variable):
        with tf.name_scope('Train'):
            with tf.variable_scope('model', reuse=None, initializer=init_variable):
                self.status = 'train'
                self.add_placeholder()
                self.forward_BIRNN_corpus()
                self.computer_crf_loss()
                self.computer_train_Adam()
                self.computer_crf_predict()

    def get_crf_test_without_pos_graph(self):
        with tf.name_scope('Test'):
            with tf.variable_scope('model', reuse=True):
                self.status = 'test'
                self.add_placeholder()
                self.forward_BIRNN_corpus()
                self.computer_crf_loss()
                self.computer_crf_predict()

    def get_crf_train_seq2seq_graph(self, init_variable):
        with tf.name_scope('Train'):
            with tf.variable_scope('model', reuse=None, initializer=init_variable):
                self.status = 'train'
                self.add_placeholder()
                self.forwards_seq2seq()
                self.computer_crf_loss()
                self.computer_train_Adam()
                self.computer_crf_predict()

    def get_crf_test_seq2seq_graph(self):
        with tf.name_scope('Test'):
            with tf.variable_scope('model', reuse=True):
                self.status = 'test'
                self.add_placeholder()
                self.forwards_seq2seq()
                self.computer_crf_loss()
                self.computer_crf_predict()

    def get_crf_train_attention_graph(self, init_variable):
        with tf.name_scope('Train'):
            with tf.variable_scope('model', reuse=None, initializer=init_variable):
                self.status = 'train'
                self.add_placeholder()
                self.forward_attention()
                self.computer_crf_loss()
                self.computer_train_Adam()
                self.computer_crf_predict()

    def get_crf_test_attention_graph(self):
        with tf.name_scope('Test'):
            with tf.variable_scope('model', reuse=True):
                self.status = 'test'
                self.add_placeholder()
                self.forward_attention()
                self.computer_crf_loss()
                self.computer_crf_predict()

    def get_crf_predict_graph(self):
        with tf.name_scope('Predict'):
            with tf.variable_scope('model', reuse=True):
                self.status = 'predict'
                self.add_placeholder()
                self.forwards()
                self.computer_crf_loss()
                self.computer_crf_predict()

    def train_run(self, session, corpus, pos, label, length):
        fetch = {'train': self.train_op,
                 'cost': self.cost,
                 'predict': self.predict}
        feed = {self.corpus_input: corpus,
                self.pos_input: pos,
                self.label_input: label,
                self.sequence_length: length}
        result = session.run(fetch, feed)
        return result['cost'], result['predict']

    def test_run(self, session, corpus, pos, label, length):
        fetch = {'cost': self.cost,
                 'predict': self.predict}
        feed = {self.corpus_input: corpus,
                self.pos_input: pos,
                self.label_input: label,
                self.sequence_length: length}
        result = session.run(fetch, feed)
        return result['cost'], result['predict']

    def predict_run(self, session, corpus, pos, length):
        fetch = {'predict': self.predict}
        feed = {self.corpus_input: [corpus],
                self.pos_input: [pos],
                self.sequence_length: [length]}
        result = session.run(fetch, feed)
        return result['predict']
