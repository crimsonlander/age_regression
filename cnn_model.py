import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, max_pool2d, softmax, fully_connected, flatten
import numpy as np
from pickle import load
from batch_gen import AgeGenderBatchGenerator


def conv_highway(X, size, activation_fn=tf.nn.relu, keep_prob=1.):
    gate = convolution2d(X, size, (3, 3), biases_initializer=tf.constant_initializer(-3), activation_fn=tf.nn.sigmoid)
    y = convolution2d(X, size, (3, 3), activation_fn=activation_fn)
    return gate * tf.nn.dropout(y, keep_prob) + (1. - gate) * X


def fully_connected_highway(X, size, activation_fn=tf.nn.relu, keep_prob=1.):
    gate = fully_connected(X, size, biases_initializer=tf.constant_initializer(-3), activation_fn=tf.nn.sigmoid)
    y = fully_connected(X, size, activation_fn=activation_fn)
    return gate * tf.nn.dropout(y, keep_prob) + (1. - gate) * X


def cnn(X, keep_prob):
    global output_filter
    X = convolution2d(X, 32, (5, 5), 2)
    X = conv_highway(X, 32)
    X = max_pool2d(X, (2, 2), 2, 'SAME')
    X = convolution2d(X, 64, (3, 3))
    X = conv_highway(X, 64)
    X = conv_highway(X, 64)
    X = max_pool2d(X, (2, 2), 2, 'SAME')
    X = convolution2d(X, 128, (3, 3))
    X = conv_highway(X, 128)
    X = conv_highway(X, 128)
    X = max_pool2d(X, (2, 2), 2, 'SAME')
    X = convolution2d(X, 256, (3, 3))
    X = conv_highway(X, 256)
    X = conv_highway(X, 256)
    X = tf.reduce_mean(X, [1, 2])
    X = fully_connected_highway(X, 256, keep_prob=keep_prob)
    y = fully_connected(X, 8, activation_fn=None)
    g = fully_connected(X, 2, activation_fn=None)

    return y, g


class CNNModel(object):
    def __init__(self):
        self.X_train, self.X_valid, self.X_test = load(open('data/X_64_64.pickle', 'rb'))
        self.y_train, self.y_valid, self.y_test = load(open('data/age.pickle', 'rb'))
        self.g_train, self.g_valid, self.g_test = load(open('data/gender.pickle', 'rb'))

        self.gen_train = AgeGenderBatchGenerator(self.X_train, self.y_train, self.g_train, 64,
                                                 (50, 50), (10, 10), (5, 5))
        self.gen_valid = AgeGenderBatchGenerator(self.X_valid, self.y_valid, self.g_valid,
                                                 200, (64, 64), (10, 10), (0, 0), augment=False)
        self.gen_test = AgeGenderBatchGenerator(self.X_test, self.y_test, self.g_test,
                                                500, (64, 64), (10, 10), (0, 0), augment=False)

        self.X = tf.placeholder(tf.float32, [None, None, None, 3])
        self.y = tf.placeholder(tf.int64, [None])
        self.g = tf.placeholder(tf.int64, [None])

        self.keep_prob = tf.placeholder(tf.float32)

        y_, g_ = cnn(self.X / 255., self.keep_prob)

        self.loss = tf.add(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_, self.y)),
                           tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(g_, self.g)),
                           name='loss')

        self.out_y = tf.arg_max(y_, 1)
        self.out_g = tf.arg_max(g_, 1)

        self.accuracy_y = tf.reduce_mean(tf.cast(tf.equal(self.y, self.out_y), tf.float32))
        self.accuracy_g = tf.reduce_mean(tf.cast(tf.equal(self.g, self.out_g), tf.float32))

        self.optimize = tf.train.AdadeltaOptimizer(0.1).minimize(self.loss)