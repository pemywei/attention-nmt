import tensorflow as tf
import data_utils
import os
import math
import sys


class Seq2SeqModel(object):
    def __init__(self, learning_rate, learning_rate_decay_factor, source_vocab_size=40000, target_vocab_size=40000, num_steps=100, num_epochs=10,
                 is_training=True):
        self.min_loss = float(sys.maxint)
        self.batch_size = 100
        self.dropout_rate = 0.5
        self.max_gradient_norm = 5
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.num_layers = 1
        self.emb_dim = 100
        self.hidden_dim = 100
        self.attention_hidden_dim = 100
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.global_step = tf.Variable(0, trainable=False)

        # placeholder of encoder_inputs, decoder_inputs, y_outputs
        self.encoder_inputs, self.decoder_inputs, self.y_outputs = self.create_placeholder()

        # source and target word embedding
        self.source_embedding = tf.get_variable("source_emb", [self.source_vocab_size, self.emb_dim])
        self.target_embedding = tf.get_variable("target_emb", [self.target_vocab_size, self.emb_dim])

        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.target_vocab_size])
        self.softmax_b = tf.get_variable("softmax_b", [self.target_vocab_size])
        self.attention_W = tf.get_variable("attention_W", [self.hidden_dim * 4, self.attention_hidden_dim])
        self.attention_U = tf.get_variable("attention_U", [self.hidden_dim * 2, self.attention_hidden_dim])
        self.attention_V = tf.get_variable("attention_V", [self.attention_hidden_dim, 1])

        self.encoder_inputs_emb = tf.nn.embedding_lookup(self.source_embedding, self.encoder_inputs)
        self.encoder_inputs_emb = tf.transpose(self.encoder_inputs_emb, [1, 0, 2])
        self.encoder_inputs_emb = tf.reshape(self.encoder_inputs_emb, [-1, self.emb_dim])
        self.encoder_inputs_emb = tf.split(0, self.num_steps, self.encoder_inputs_emb)

        self.decoder_inputs_emb = tf.nn.embedding_lookup(self.target_embedding, self.decoder_inputs)
        self.decoder_inputs_emb = tf.transpose(self.decoder_inputs_emb, [1, 0, 2])
        self.decoder_inputs_emb = tf.reshape(self.decoder_inputs_emb, [-1, self.emb_dim])
        self.decoder_inputs_emb = tf.split(0, self.num_steps, self.decoder_inputs_emb)

        # lstm cell
        self.enc_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        self.enc_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        self.dec_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim * 2)

        # dropout
        if is_training:
            self.enc_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(self.enc_lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
            self.enc_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(self.enc_lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))
            self.dec_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(self.dec_lstm_cell, output_keep_prob=(1 - self.dropout_rate))

        # get the length of each sample
        self.source_length = tf.reduce_sum(tf.sign(self.encoder_inputs), reduction_indices=1)
        self.source_length = tf.cast(self.source_length, tf.int32)
        self.target_length = tf.reduce_sum(tf.sign(self.decoder_inputs), reduction_indices=1)
        self.target_length = tf.cast(self.target_length, tf.int32)

        # encode and decode
        enc_outputs, enc_state = self.encode(self.enc_lstm_cell_fw, self.enc_lstm_cell_bw)
        if is_training:
            self.dec_outputs = self.decode(self.dec_lstm_cell, enc_state, enc_outputs)
        else:
            self.dec_outputs = self.decode(self.dec_lstm_cell, enc_state, enc_outputs, self.loop_function)
        # softmax
        self.outputs = tf.reshape(tf.concat(1, self.dec_outputs), [-1, self.hidden_dim * 2])
        self.logits = tf.add(tf.matmul(self.outputs, self.softmax_w), self.softmax_b)
        self.prediction = tf.nn.softmax(self.logits)

        self.y_output = tf.reshape(self.y_outputs, [-1])
        self.y_output = tf.one_hot(self.y_output, depth=self.target_vocab_size, on_value=1.0, off_value=0.0)
        self.cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y_output))

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        gradients = tf.gradients(self.cross_entropy_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = self.optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.all_variables())

    def create_placeholder(self):
        encoder_input_pl = tf.placeholder(tf.int64, [None, self.num_steps])
        decoder_input_pl = tf.placeholder(tf.int64, [None, self.num_steps])
        y_output_pl = tf.placeholder(tf.int64, [None, self.num_steps])
        return encoder_input_pl, decoder_input_pl, y_output_pl

    def encode(self, cell_fw, cell_bw):
        enc_outputs, output_state_fw, output_state_bw = tf.nn.bidirectional_rnn(
            cell_fw,
            cell_bw,
            self.encoder_inputs_emb,
            dtype=tf.float32,
            sequence_length=self.source_length
        )
        enc_state = tf.concat(1, [output_state_fw, output_state_bw])
        return enc_outputs, enc_state

    def attention(self, prev_state, enc_outputs):
        """
        Attention model for Neural Machine Translation
        :param prev_state: the decoder hidden state at time i-1
        :param enc_outputs: the encoder outputs, a length 'T' list.
        """
        e_i = []
        c_i = []
        for output in enc_outputs:
            atten_hidden = tf.tanh(tf.add(tf.matmul(prev_state, self.attention_W), tf.matmul(output, self.attention_U)))
            e_i_j = tf.matmul(atten_hidden, self.attention_V)
            e_i.append(e_i_j)
        e_i = tf.concat(1, e_i)
        # e_i = tf.exp(e_i)
        alpha_i = tf.nn.softmax(e_i)
        alpha_i = tf.split(1, self.num_steps, alpha_i)
        for alpha_i_j, output in zip(alpha_i, enc_outputs):
            c_i_j = tf.mul(alpha_i_j, output)
            c_i.append(c_i_j)
        c_i = tf.reshape(tf.concat(1, c_i), [-1, self.num_steps, self.hidden_dim * 2])
        c_i = tf.reduce_sum(c_i, 1)
        return c_i

    def decode(self, cell, init_state, enc_outputs, loop_function=None):
        outputs = []
        prev = None
        state = init_state
        for i, inp in enumerate(self.decoder_inputs_emb):

            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            c_i = self.attention(state, enc_outputs)
            inp = tf.concat(1, [inp, c_i])
            output, state = cell(inp, state)
            # print output.eval()
            outputs.append(output)
            if loop_function is not None:
                prev = output
        return outputs

    def loop_function(self, prev, _):
        """
        :param prev: the output of t-1 time
        :param _:
        :return: the embedding of t-1 output
        """
        prev = tf.add(tf.matmul(prev, self.softmax_w), self.softmax_b)
        prev_sympol = tf.arg_max(prev, 1)

        emb_prev = tf.nn.embedding_lookup(self.target_embedding, prev_sympol)
        return emb_prev

    def train(self, sess, save_path, train_set, val_set):
        num_iterations = int(math.ceil(1.0 * len(train_set) / self.batch_size))
        print("Number of iterations: %d" % num_iterations)
        previous_losses = []
        for epoch in range(self.num_epochs):
            print("current epoch: %d" % epoch)
            for iteration in range(num_iterations):
                batch_encoder_inputs, batch_decoder_inputs, batch_y_outputs = \
                    data_utils.nextBatch(train_set,
                                         start_index=iteration * self.batch_size,
                                         batch_size=self.batch_size)
                _, loss_train = \
                    sess.run(
                        [
                            self.updates,
                            self.cross_entropy_loss,
                        ],
                        feed_dict={
                            self.encoder_inputs: batch_encoder_inputs,
                            self.decoder_inputs: batch_decoder_inputs,
                            self.y_outputs: batch_y_outputs
                        })

                if len(previous_losses) > 2 and loss_train > max(previous_losses[-3:]):
                    sess.run(self.learning_rate_decay_op)
                previous_losses.append(loss_train)

                if iteration % 10 == 0:
                    # ppl = math.exp(loss_train)
                    print("    iteration: %d, train loss: %5f" % (iteration, loss_train))
                if iteration % 100 == 0:
                    batch_encoder_val, batch_decoder_val, batch_y_val = \
                        data_utils.nextRandomBatch(val_set, batch_size=self.batch_size)
                    loss_val = \
                        sess.run(
                            self.cross_entropy_loss,
                            feed_dict={
                                self.encoder_inputs: batch_encoder_val,
                                self.decoder_inputs: batch_decoder_val,
                                self.y_outputs: batch_y_val
                            })
                    # ppl_val = math.exp(loss_val)
                    print("    iteration: %d, valid loss: %5f" % (iteration, loss_val))

                    if loss_val < self.min_loss:
                        self.min_loss = loss_val
                        checkpoint_path = os.path.join(save_path, "translate.ckpt")
                        print("Saving model in %s" % checkpoint_path)
                        self.saver.save(sess, checkpoint_path, global_step=self.global_step)
                        # self.saver.save(sess, save_path)
                        print("saved the best model with loss: %.5f" % loss_val)

    def test(self, sess, token_ids):
        # We decode one sentence at a time.
        token_ids = data_utils.padding(token_ids)
        target_ids = data_utils.padding([data_utils.GO_ID])
        y_ids = data_utils.padding([data_utils.EOS_ID])
        encoder_inputs, decoder_inputs, _ = data_utils.nextRandomBatch([(token_ids, target_ids, y_ids)], batch_size=1)
        prediction = sess.run(self.prediction, feed_dict={
            self.encoder_inputs: encoder_inputs,
            self.decoder_inputs: decoder_inputs
        })
        pred_max = tf.arg_max(prediction, 1)
        # prediction = tf.split(0, self.num_steps, prediction)
        # # This is a greedy decoder - outputs are just argmaxes of output_logits.
        # outputs = [int(np.argmax(predict)) for predict in prediction]
        # # If there is an EOS symbol in outputs, cut them at that point.
        # if data_utils.EOS_ID in outputs:
        #     outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        return pred_max.eval()
