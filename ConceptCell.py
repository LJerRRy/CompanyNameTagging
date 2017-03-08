import tensorflow as tf

from ReadLocationNames import locations


class ConceptCell:
    input_dimension = 8  # SHA-256

    rbf_num = len(locations)
    alpha = 1.0
    beta = 1.0
    theta = 1.0
    model_path = './model/'
    saver = None
    learning_rate = 0.1
    epoch_count = 0

    x = tf.placeholder(tf.int64, shape=[None, 1, input_dimension])
    y_ = tf.placeholder(tf.int64, shape=[None, 1])
    c0 = tf.zeros([rbf_num, input_dimension], dtype=tf.float64)
    # c0 = None
    c = tf.Variable(tf.zeros([rbf_num, input_dimension], dtype=tf.float64))  # TODO: initialization strategy
    r = tf.Variable(tf.ones([1, rbf_num], dtype=tf.float64))  # TODO: initialization strategy & division-by-zero
    w = tf.ones([rbf_num, 1], dtype=tf.float64)  # Could be variables
    rbfs = None
    rbf_center_distance = None
    y = None
    loss = None

    sess = None
    merged_summary = None
    summary_writer = None

    def __init__(self, model_name):
        x_ = tf.tile(self.x, [1, self.rbf_num, 1])

        self.rbfs = tf.to_double(x_) - self.c          # Broadcasting feature. Cool!
        self.rbfs = tf.square(self.rbfs)
        self.rbfs = tf.reduce_sum(self.rbfs, 2)
        # self.rbf_center_distance = tf.reduce_sum(self.rbfs, 1)
        self.rbfs = tf.multiply(self.rbfs, tf.reciprocal(self.r))
        self.rbfs = -self.rbfs
        self.rbfs = tf.exp(self.rbfs)

        self.y = tf.tanh(tf.matmul(self.rbfs, self.w))          # No bias? Only one output?

        self.loss = tf.reduce_mean(tf.square(self.y - tf.to_double(self.y_))) \
            + self.alpha * tf.reduce_sum(self.beta * tf.sigmoid(tf.square(self.c0 - self.c))) \
            + self.theta * tf.reduce_sum(tf.square(self.r))

        self.model_path += model_name
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        with tf.name_scope('location_cell'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('loss_histogram', self.loss)

        self.merged_summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('./log/', self.sess.graph)

    def save_params(self):
        self.saver.save(self.sess, self.model_path)
        print('Model saved.')

    def restore_params(self):
        self.saver.restore(self.sess, self.model_path)
        print('Model restored.')

    def train(self, X, Y_):
        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.c = tf.assign(self.c, tf.to_double(tf.tile(X[0], [self.rbf_num, 1])), name='c_init')
        self.c0 = tf.tile(X[0], [self.rbf_num, 1], name='c0_init')
        # print(self.sess.run(self.c))

        with self.sess.as_default():
            print('Total loss before training: %i' % self.loss.eval({self.x: X, self.y_: Y_}))

        for i in range(self.epoch_count):
            print('epoch: %i' % i)
            if i % 10 == 0:
                _, summary = self.sess.run([train_step, self.merged_summary], {self.x: X, self.y_: Y_})
                self.summary_writer.add_summary(summary, i)

            else:
                self.sess.run(train_step, {self.x: X, self.y_: Y_})

        self.save_params()
        with self.sess.as_default():
            print('Total training loss: %i' % self.loss.eval({self.x: X, self.y_: Y_}))

    def resume_training(self, X, Y_):
        self.restore_params()
        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        with self.sess.as_default():
            print('Total loss on last training session: %i' % self.loss.eval({self.x: X, self.y_: Y_}))
            for i in range(self.epoch_count):
                print('epoch: %i' % i)
                if i % 10 == 0:
                    _, summary = self.sess.run([train_step, self.merged_summary], {self.x: X, self.y_: Y_})
                    self.summary_writer.add_summary(summary, i)

                else:
                    self.sess.run(train_step, {self.x: X, self.y_: Y_})

                # print(self.loss.eval({self.x: X, self.y_: Y_}))

            self.save_params()
            print('Total training loss: %i' % self.loss.eval({self.x: X, self.y_: Y_}))
