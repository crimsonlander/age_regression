import tensorflow as tf
from operator import itemgetter
from time import time
import numpy as np
from multiprocessing import Process, Pipe, Event


class BatchLoader(Process):
    def __init__(self, batch_function):
        Process.__init__(self)
        self.pipe1, self.pipe2 = Pipe(True)
        self.exit = Event()
        self.batch_function = batch_function

    def run(self):
        while not self.exit.is_set():
            self.pipe1.send(self.batch_function())

    def terminate(self):
        self.exit.set()


class Trainer(object):
    def __init__(self, optimizer,
                 train_inputs, batch_func_train,
                 valid_inputs=None, batch_func_valid=None,
                 metrics_train=(), metrics_valid=(), valid_freq=100, save_freq=1000, log_name='log',
                 save_name=None, load_name=None):

        self.batch_func_train = batch_func_train
        self.batch_func_valid = batch_func_valid
        self.inputs = train_inputs
        self.optimizer = optimizer
        metrics_train = metrics_train if isinstance(metrics_train, (list, tuple)) else list(metrics_train.items())
        self.metric_train_names = list(map(itemgetter(0), metrics_train))
        self.metric_train_tensors = list(map(itemgetter(1), metrics_train))

        metrics_valid = metrics_valid if isinstance(metrics_valid, (list, tuple)) else list(metrics_valid.items())
        self.metric_valid_names = list(map(itemgetter(0), metrics_valid))
        self.metric_valid_tensors = list(map(itemgetter(1), metrics_valid))

        self.valid_freq = valid_freq
        self.log_name = log_name
        self.valid_inputs = valid_inputs
        self.save_freq = save_freq
        self.save_name = save_name
        self.load_name = load_name

    def train(self, num_steps):
        saver = tf.train.Saver()
        print("Train started for %d steps" % num_steps)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        if self.load_name is not None:
            saver.restore(sess, "models/%s.ckpt" % self.load_name)

        p = BatchLoader(self.batch_func_train)
        p.start()

        for i in range(num_steps):

            try:
                batch = p.pipe2.recv()

                feed_dict = {self.inputs[j]: batch[j] for j in range(len(self.inputs))}
                t1 = time()
                sess.run(self.optimizer, feed_dict=feed_dict)
                t2 = time()
                if i % self.valid_freq == 0:
                    log = open("logs/%s.log" % self.log_name, 'a')
                    print("Step: " + str(i))
                    log.write("Step: " + str(i) + "\n")
                    print("Learning step time", t2 - t1)
                    metric_values = sess.run(self.metric_train_tensors, feed_dict)

                    for j in range(len(metric_values)):
                        log.write('train ' + self.metric_train_names[j] + ' ' + str(metric_values[j]) + '\n')
                        print('train ' + self.metric_train_names[j] + ' ' + str(metric_values[j]))

                    if self.valid_inputs is not None:
                        validation = self.batch_func_valid()
                        feed_dict = {self.valid_inputs[i]: validation[i] for i in range(len(self.inputs))}

                        metric_values = sess.run(self.metric_valid_tensors, feed_dict)

                        for j in range(len(metric_values)):
                            log.write('validation ' + self.metric_valid_names[j] + ' ' + str(metric_values[j]) + '\n')
                            print('validation ' + self.metric_valid_names[j] + ' ' + str(metric_values[j]))
                    log.write("---\n")
                    print("---")
                    log.close()

                if self.save_name is not None and i % self.save_freq == 0:
                    saver = tf.train.Saver()
                    saver.save(sess, "checkpoints/%s_%d.ckpt" % (self.save_name, i))

            except Exception as e:
                print(e)
                if self.save_name is not None:
                    saver.save(sess, "checkpoints/%s_%d.ckpt" % (self.save_name, i))
                break

        if self.save_name is not None:
            saver.save(sess, "models/%s.ckpt" % (self.save_name,))

        p.terminate()
        p.pipe2.recv()