import time
import tensorflow as tf
import data_utils
import seq2seq
import os
import sys
import jieba

tf.app.flags.DEFINE_integer("cn_vocab_size", 40000, "Chinese vocabulary size.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_string("data_dir", "data/", "Data directory.")
tf.app.flags.DEFINE_string("train_dir", "model/", "Train directory.")
tf.app.flags.DEFINE_integer("epochs", 10, "The number of epoch.")

FLAGS = tf.app.flags.FLAGS

num_steps = 100

start_time = time.time()


def create_model(sess, is_training=True):
    model = seq2seq.Seq2SeqModel(FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
                                 FLAGS.cn_vocab_size, FLAGS.en_vocab_size, num_steps=num_steps, num_epochs=FLAGS.epochs,
                                 is_training=is_training)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)

    else:
        print("Created model with fresh parameters.")
        sess.run(tf.global_variables_initializer())
    return model


def main(_):
    print("Loading vocabulary")
    cn_vocab_path = os.path.join(FLAGS.data_dir, "source_vocab.txt")
    en_vocab_path = os.path.join(FLAGS.data_dir, "target_vocab.txt")

    cn_vocab, _ = data_utils.initialize_vocabulary(cn_vocab_path)
    _, rev_en_vocab = data_utils.initialize_vocabulary(en_vocab_path)

    print("Building model...")
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        model = create_model(sess, False)
        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            seg_list = jieba.lcut(sentence.strip())
            print(" ".join(seg_list))
            token_ids = [cn_vocab.get(w.encode(encoding="utf-8"), data_utils.UNK_ID) for w in seg_list]
            print(token_ids)
            outputs = model.test(sess, token_ids)
            output = " ".join([tf.compat.as_str(rev_en_vocab[output]) for output in outputs])
            print(output.capitalize())
            print("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


if __name__ == "__main__":
    tf.app.run()
