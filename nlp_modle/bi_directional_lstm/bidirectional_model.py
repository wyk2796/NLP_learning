# coding:utf-8
import tensorflow as tf


class BidirectionalModel(object):
    def __init__(self, config):
        self.config = config
        self.hidden_size = config.hidden_size
        self.learning_rate = config.learning_rate

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

        self.logits = None
        self.cost = None
        self.train_op = None
        self.predict = None
        self.precision = None

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


        """
        transforming the dimensional of input_data
        fist transpose the dimensional of input from [batch_size, step_num, vector] to [step_num, batch_size, vector]
        second step, flatting the first and second dimensional that become [step_num * batch_size, vector], then split
        the matrix by batch_size.

        """
        # input_data = tf.split(tf.reshape(tf.transpose(input_data, [1, 0, 2]), [-1, self.hidden_size]), self.step_num)
        # pos_data = tf.split(tf.reshape(tf.transpose(pos_data, [1, 0, 2]), [-1, self.hidden_size]), self.step_num)

        """
        encode bidirectional_model
        """

        print(input_data.shape)
        _, (fw, bw) = tf.nn.bidirectional_dynamic_rnn(self._lstm_cell(),
                                                      self._lstm_cell(),
                                                      input_data,
                                                      dtype=tf.float32,
                                                      # time_major=True,
                                                      sequence_length=self.sequence_length,
                                                      scope='encode')

        """
        decode bidirectional_model
        """

        (out_result_fw, out_result_bw), _ = tf.nn.bidirectional_dynamic_rnn(self._lstm_cell(),
                                                        self._lstm_cell(),
                                                        pos_data,
                                                        initial_state_fw=fw,
                                                        initial_state_bw=bw,
                                                        dtype=tf.float32,
                                                        # time_major=True,
                                                        sequence_length=self.sequence_length,
                                                        scope='decode')
        result = tf.concat([out_result_fw, out_result_bw], axis=2)

        with tf.variable_scope('output_layer', reuse=None):
            u_w = tf.get_variable('U_weight', shape=[self.hidden_size * 2, self.label_vocabulary_size])
            u_b = tf.get_variable('U_bias', shape=[self.step_num, self.label_vocabulary_size])
        logits_list = [tf.matmul(result[i], u_w) + u_b for i in range(self.batch_size)]
        self.logits = tf.concat(logits_list, axis=0)

    def computer_cost(self):
        label = tf.reshape(tf.one_hot(self.label_input, self.label_vocabulary_size), [-1, self.label_vocabulary_size])
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                              labels=label))
        self.cost = self.cost / self.batch_size

    def computer_train(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def computer_predict(self):
        self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict_output')
        self.predict = tf.reshape(self.predict, shape=[self.batch_size, self.step_num])

    def computer_precision_rate(self):
        batch_seq_mask = tf.sequence_mask(self.sequence_length, maxlen=self.step_num)
        correct_predict = tf.equal(tf.cast(self.predict, tf.int32), self.label_input)
        reduce_value = tf.boolean_mask(correct_predict, batch_seq_mask)
        print(tf.reshape(correct_predict, [-1]).shape)
        print(tf.reshape(reduce_value, [-1]).shape)
        self.precision = tf.reduce_mean(tf.cast(reduce_value, tf.float32))

    def get_train_graph(self, init_variable):
        with tf.name_scope('Train'):
            with tf.variable_scope('model', reuse=None, initializer=init_variable):
                self.status = 'train'
                self.add_placeholder()
                self.forwards()
                self.computer_cost()
                self.computer_train()
                self.computer_predict()
                self.computer_precision_rate()

    def get_test_graph(self):
        with tf.name_scope('Predict'):
            with tf.variable_scope('model', reuse=True):
                self.status = 'predict'
                self.add_placeholder()
                self.forwards()
                self.computer_cost()
                self.computer_predict()
                self.computer_precision_rate()

    def train_run(self, session, corpus, pos, label, length):
        fetch = {'train': self.train_op, 'cost': self.cost, 'precision': self.precision}
        feed = {self.corpus_input: corpus,
                self.pos_input: pos,
                self.label_input: label,
                self.sequence_length: length}
        result = session.run(fetch, feed)
        return result['cost'], result['precision']

    def test_run(self, session, corpus, pos, label, length):
        fetch = {'cost': self.cost, 'precision': self.precision}
        feed = {self.corpus_input: corpus,
                self.pos_input: pos,
                self.label_input: label,
                self.sequence_length: length}
        result = session.run(fetch, feed)
        return result['cost'], result['precision']
