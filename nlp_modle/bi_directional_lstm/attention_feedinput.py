import tensorflow as tf

def attention_seq2seq(self, from_fw_single_cell, mid_fw_single_cell, sources, inputs, dtype, devices=None):
    # a = softmax( a_v * tanh(a_w_source * h_source + a_w_target * h_target + a_b))
    # context = a * h_source
    # h_target_attent = tanh(h_w_context * context + h_w_target * h_target + h_b)
    # feed_input: x = fi_w_x * decoder_input + fi_w_att * prev_h_target_attent) + fi_b



    with tf.variable_scope("attention_seq2seq"):
        # with tf.device(devices[1]):
        init_state = from_fw_single_cell.zero_state(self.batch_size, dtype)

        # 10parameters
        self.a_w_source = tf.get_variable("a_w_source", [self.size, self.size], dtype=dtype)
        self.a_w_target = tf.get_variable('a_w_target', [self.size, self.size], dtype=dtype)
        self.a_b = tf.get_variable('a_b', [self.size], dtype=dtype)

        self.a_v = tf.get_variable('a_v', [self.size], dtype=dtype)

        self.h_w_context = tf.get_variable("h_w_context", [self.size, self.size], dtype=dtype)
        self.h_w_target = tf.get_variable("h_w_target", [self.size, self.size], dtype=dtype)
        self.h_b = tf.get_variable('h_b', [self.size], dtype=dtype)

        self.fi_w_x = tf.get_variable("fi_w_x", [self.size, self.size], dtype=dtype)
        self.fi_w_att = tf.get_variable("fi_w_att", [self.size, self.size], dtype=dtype)
        self.fi_b = tf.get_variable('fi_b', [self.size], dtype=dtype)

        # todo xhh 应该是当前bucket的长度
        source_length = len(sources)

        with tf.variable_scope("encoder"):
            # encoder lstm
            encoder_outputs, encoder_state = tf.contrib.rnn.static_rnn(from_fw_single_cell, sources,
                                                                       initial_state=init_state)

            # combine all source hts to top_states [batch_size, source_length, hidden_size]
            # todo hs三维矩阵
            top_states = [tf.reshape(h, [-1, 1, self.size]) for h in encoder_outputs]
            top_states = tf.concat(top_states, 1)

            # calculate a_w_source * h_source
            # todo hs变成四维 a_w_source变成四维
            top_states_4 = tf.reshape(top_states, [-1, source_length, 1, self.size])
            a_w_source_4 = tf.reshape(self.a_w_source, [1, 1, self.size, self.size])
            top_states_transform_4 = tf.nn.conv2d(top_states_4, a_w_source_4, [1, 1, 1, 1],
                                                  'SAME')  # [batch_size, source_length, 1, hidden_size]

        # query =ht return ht~
        # todo 给了ht 输出ct
        def get_context(query):
            # query : [batch_size, hidden_size]
            # return h_t_att : [batch_size, hidden_size]

            # a_w_target * h_target + a_b
            query_transform_2 = tf.add(tf.matmul(query, self.a_w_target), self.a_b)
            query_transform_4 = tf.reshape(query_transform_2, [-1, 1, 1, self.size])  # [batch_size,1,1,hidden_size]

            # a = softmax( a_v * tanh(...))
            s = tf.reduce_sum(self.a_v * tf.tanh(top_states_transform_4 + query_transform_4),
                              [2, 3])  # [batch_size, source_length]
            a = tf.nn.softmax(s)

            # context = a * h_source
            context = tf.reduce_sum(tf.reshape(a, [-1, source_length, 1, 1]) * top_states_4, [1, 2])

            return context

        with tf.variable_scope("decoder"):
            state = encoder_state
            ht = encoder_outputs[-1]

            prev_h_att = tf.zeros_like(ht)

            outputs = []

            for i in range(len(inputs)):
                decoder_input = inputs[i]

                # x = fi_w_x * decoder_input + fi_w_att * prev_h_target_attent) + fi_b
                x = tf.add(tf.add(tf.matmul(decoder_input, self.fi_w_x),tf.matmul(prev_h_att, self.fi_w_att)), self.fi_b)

                # decoder lstm
                decoder_output, state = mid_fw_single_cell(x, state)

                context = get_context(decoder_output)

                # h_target_attent = tanh(h_w_context * context + h_w_target * h_target + h_b)
                h_att = tf.tanh(
                    tf.add(tf.add(tf.matmul(decoder_output, self.h_w_target), tf.matmul(context, self.h_w_context)),
                           self.h_b))

                prev_h_att = h_att

                outputs.append(h_att)

        return outputs, state


