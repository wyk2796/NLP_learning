# coding:utf-8
import tensorflow as tf

def attention_layer(con_corpus, con_pos, batch_size, step_num, hidden_size):

    with tf.variable_scope('attention'):
        a_w_source = tf.get_variable("a_w_source", [hidden_size * 2, hidden_size * 2])
        a_w_target = tf.get_variable('a_w_target', [hidden_size * 2, hidden_size * 2])
        a_b = tf.get_variable('a_b', [hidden_size * 2])

        a_v = tf.get_variable('a_v', [hidden_size * 2])

        h_w_context = tf.get_variable("h_w_context", [hidden_size * 2, hidden_size * 2])
        h_w_target = tf.get_variable("h_w_target", [hidden_size * 2, hidden_size * 2])
        h_b = tf.get_variable('h_b', [hidden_size * 2])

    top_states = [tf.reshape(h, [-1, 1, hidden_size * 2]) for h in tf.split(tf.transpose(con_corpus, [1, 0, 2]), step_num)]
    top_states = tf.concat(top_states, 1)
    top_states_4 = tf.reshape(top_states, [-1, step_num, 1, hidden_size * 2])
    a_w_source_4 = tf.reshape(a_w_source, [1, 1, hidden_size * 2, hidden_size * 2])
    top_states_transform_4 = tf.nn.conv2d(top_states_4, a_w_source_4, [1, 1, 1, 1], 'SAME')

    def get_context(query):
        # query : [batch_size, hidden_size]
        # return h_t_att : [batch_size, hidden_size]

        # a_w_target * h_target + a_b
        query_transform_2 = tf.add(tf.matmul(query, a_w_target), a_b)
        query_transform_4 = tf.reshape(query_transform_2, [-1, 1, 1, hidden_size * 2])  # [batch_size,1,1,hidden_size]

        # a = softmax( a_v * tanh(...))
        s = tf.reduce_sum(a_v * tf.tanh(top_states_transform_4 + query_transform_4),
                          [2, 3])  # [batch_size, source_length]
        a = tf.nn.softmax(s)

        # context = a * h_source
        context = tf.reduce_sum(tf.reshape(a, [-1, step_num, 1, 1]) * top_states_4, [1, 2])

        return context

    outputs = []
    for pos in tf.split(tf.transpose(con_pos, [1, 0, 2]), step_num):
        pos = tf.reshape(pos, [batch_size, hidden_size * 2])
        context = get_context(pos)

        h_att = tf.tanh(
            tf.add(tf.add(tf.matmul(pos, h_w_target), tf.matmul(context, h_w_context)), h_b))

        ##prev_h_att = h_att

        outputs.append(h_att)
    return outputs
