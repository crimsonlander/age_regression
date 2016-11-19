import tensorflow as tf
from train import Trainer


def test_model(model, name):

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    saver.restore(sess, "models/%s.ckpt" % name)

    acc = 0

    print('Testing...')
    for i in range(10):
        batch = model.gen_test.get_supervised_batch()
        feed_dict = dict()

        feed_dict[model.X] = batch[0]
        feed_dict[model.y] = batch[1]
        feed_dict[model.keep_prob] = 1.

        acc += sess.run([model.accuracy_y], feed_dict=feed_dict)[0]

    print("%s accuracy: %f" % (name, acc / 10))
    return acc / 10


def train_model(model, name, num_steps):
    trainer = Trainer(model.optimize, [model.X, model.y, model.g, model.keep_prob],
                      lambda: model.gen_train.get_supervised_batch() + (0.5,),
                      [model.X, model.y, model.g, model.keep_prob],
                      lambda: model.gen_valid.get_supervised_batch() + (1.,),
                      {'loss': model.loss, 'age accuracy': model.accuracy_y, 'gender accuracy': model.accuracy_g},
                      {'loss': model.loss, 'age accuracy': model.accuracy_y, 'gender accuracy': model.accuracy_g},
                      save_name=name, save_freq=1000, log_name=name)

    trainer.train(num_steps)