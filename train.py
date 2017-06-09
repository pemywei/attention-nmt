import time
import data_utils
import tensorflow as tf
import seq2seq
import os

tf.app.flags.DEFINE_string("data_dir", "data/", "Data directory.")
tf.app.flags.DEFINE_string("train_dir", "model/", "Train directory.")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_integer("epochs", 10, "The number of epoch.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "The number of epoch.")
tf.app.flags.DEFINE_integer("cn_vocab_size", 40000, "Chinese vocabulary size.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")

FLAGS = tf.app.flags.FLAGS

num_steps = 100

start_time = time.time()
# Prepare data.
print("Preparing data in %s" % FLAGS.data_dir)
source_path = os.path.join(FLAGS.data_dir, "source_token_ids.txt")
target_path = os.path.join(FLAGS.data_dir, "target_token_ids.txt")

data_set = data_utils.read_data(source_path, target_path)
train_set, val_set = data_utils.get_train_validation_set(data_set)


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
    print("Building model...")
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        model = create_model(sess, True)
        print("Training model")
        model.train(sess, FLAGS.train_dir, train_set, val_set, FLAGS.steps_per_checkpoint)
        print("final best loss is: %f" % model.min_loss)

        end_time = time.time()
        print("time used %f(hour)" % ((end_time - start_time) / 3600))


if __name__ == "__main__":
    tf.app.run()
