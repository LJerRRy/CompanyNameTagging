from ReadLocationNames import locations
import tensorflow as tf


class ConceptCell:
    input_dimension = 8  # SHA-256

    rbf_num = len(locations)
    alpha = 1.0
    beta = 1.0
    theta = 1.0
    model_path = './model/'
    saver = None
    epoch_count = 1000

    x = tf.placeholder(tf.int64, shape=[None, 1, input_dimension])
    y_ = tf.placeholder(tf.int64, shape=[None, 1])
    c0 = tf.zeros([rbf_num, input_dimension])
    c = tf.Variable(tf.zeros([rbf_num, input_dimension]))  # TODO: initialization strategy
    r = tf.Variable(tf.ones([1, rbf_num]))  # TODO: initialization strategy & division-by-zero
    w = tf.ones([rbf_num, 1])  # Could be variables
    rbfs = None
    y = None
    loss = None
    sess = None

    def __init__(self, model_name):
        x_ = tf.tile(self.x, [1, self.rbf_num, 1])

        self.rbfs = tf.to_float(x_) - self.c          # Broadcasting feature. Cool!
        self.rbfs = tf.square(self.rbfs)
        self.rbfs = tf.reduce_sum(self.rbfs, 2)
        self.rbfs = tf.multiply(self.rbfs, tf.reciprocal(self.r))
        self.rbfs = -self.rbfs
        self.rbfs = tf.exp(self.rbfs)

        self.y = tf.tanh(tf.matmul(self.rbfs, self.w))          # No bias? Only one output?

        self.loss = tf.reduce_mean(tf.square(self.y - tf.to_float(self.y_))) \
            + self.alpha * tf.reduce_sum(self.beta * tf.sigmoid(tf.square(self.c0 - self.c))) \
            + self.theta * tf.reduce_sum(tf.square(self.r))

        self.model_path += model_name
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def save_params(self):
        self.saver.save(self.sess, self.model_path)
        print('Model saved.')

    def restore_params(self):
        self.saver.restore(self.sess, self.model_path)
        print('Model restored.')

    def train(self, X, Y_):
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for i in range(self.epoch_count):
            print('epoch: %i' % i)
            self.sess.run(train_step, {self.x: X, self.y_: Y_})

        self.save_params()
        with self.sess.as_default():
            print('Total training loss: %i' % self.loss.eval({self.x: X, self.y_: Y_}))

