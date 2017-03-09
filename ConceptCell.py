import tensorflow as tf

from ReadLocationNames import locations


class ConceptCell:
    input_dimension = 8  # SHA-256

    rbf_num = len(locations)
    alpha = 0
    beta = 1.0
    theta = 0
    scaling_factor = 10000
    model_path = None
    saver = None
    learning_rate = 0.1
    epoch_count = 1

    x = tf.placeholder(tf.int64, shape=[None, 1, input_dimension], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, 1], name='y_')
    # c0 = tf.zeros([rbf_num, input_dimension], dtype=tf.float64, name='c0')
    c0 = None
    # c = tf.Variable(tf.zeros([rbf_num, input_dimension], dtype=tf.float64), name='c')
    c = None
    # r = tf.Variable(tf.ones([1, rbf_num], dtype=tf.float64), name='r')
    r = None
    # r = tf.ones([1, rbf_num], dtype=tf.float64, name='r')
    # w = tf.ones([rbf_num, 1], dtype=tf.float64, name='w')  # Could be variables
    w = None
    rbfs = None
    rbf_center_distance = None
    y = None
    loss = None

    sess = None
    merged_summary = None
    summary_writer = None

    def __init__(self, model_name):
        self.model_path = './model/' + model_name
        self.sess = tf.Session()

    def define_model(self):
        x_ = tf.tile(self.x, [1, self.rbf_num, 1], name='expand_x')

        self.rbfs = tf.to_double(x_) - self.c          # Broadcasting feature. Cool!
        self.rbfs = tf.square(self.rbfs)
        self.rbfs = tf.reduce_sum(self.rbfs, 2)
        # self.rbf_center_distance = tf.reduce_sum(self.rbfs, 1)
        self.rbfs = tf.multiply(self.rbfs, tf.reciprocal(self.r))
        self.rbfs = -self.rbfs
        self.rbfs = tf.exp(self.rbfs)

        self.y = tf.tanh(tf.matmul(self.rbfs, self.w), name='squeeze_output')          # No bias? Only one output?

        # Some issues for loss
        # self.loss = tf.reduce_mean(tf.square(self.y - tf.to_double(self.y_))) \
        #     + self.alpha * tf.reduce_sum(self.beta * tf.sigmoid(tf.square(self.c0 - self.c))) \
        #     + self.theta * tf.reduce_sum(tf.square(self.r))

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
        self.rbf_num = len(X)
        self.c = tf.Variable(tf.zeros([self.rbf_num, self.input_dimension], dtype=tf.float64), name='c')
        self.r = tf.Variable(tf.ones([1, self.rbf_num], dtype=tf.float64), name='r')
        self.w = tf.ones([self.rbf_num, 1], dtype=tf.float64, name='w')  # Could be variables

        c0_ = []
        for pt in X:
            c0_.append(pt[0])
        self.c0 = tf.constant(c0_, dtype=tf.float64, name='init_c0')
        self.c = tf.assign(self.c, c0_, name='init_c')

        self.define_model()

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train_step = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # with self.sess.as_default():
        #     self.c = tf.assign(self.c, tf.to_double(tf.tile(X[0], [self.rbf_num, 1])), name='init_c')
        #     self.c0 = tf.tile(X[0], [self.rbf_num, 1], name='init_c0')
        # self.sess.run([self.c, self.c0])        # Must run once
        #
        # with self.sess.as_default():
        #     print('Total loss before training: %f' % self.loss.eval({self.x: X, self.y_: Y_}))
        #
        # for i in range(self.epoch_count):
        #     print('epoch: %i' % i)
        #     if (i+1) % 1 == 0:
        #         _, summary, loss = self.sess.run([train_step, self.merged_summary, self.loss], {self.x: X, self.y_: Y_})
        #         self.summary_writer.add_summary(summary, i)
        #         print('Training loss: %f' % loss)
        #         grads_and_vars = optimizer.compute_gradients(self.loss)         # Naught grad for c, r!
        #         for gv in grads_and_vars:
        #             # print(str(self.sess.run(gv[0], {self.x: X, self.y_: Y_})) + ' - ' + gv[1].name)
        #             print('Sum of gradients of variable %s: %f' % (gv[1].name, self.sess.run(tf.reduce_sum(gv[0]), {self.x: X, self.y_: Y_})))
        #
        #         # print(self.sess.run(self.c) == self.sess.run(self.c0))
        #
        #     else:
        #         self.sess.run(train_step, {self.x: X, self.y_: Y_})

        self.save_params()
        with self.sess.as_default():
            print('Total training loss: %f' % self.loss.eval({self.x: X, self.y_: Y_}))
        self.outputs(X, Y_)

    def resume_training(self, X, Y_):
        self.restore_params()
        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        with self.sess.as_default():
            print('Total loss on last training session: %f' % self.loss.eval({self.x: X, self.y_: Y_}))
            for i in range(self.epoch_count):
                print('epoch: %i' % i)
                if i % 10 == 0:
                    _, summary = self.sess.run([train_step, self.merged_summary], {self.x: X, self.y_: Y_})
                    self.summary_writer.add_summary(summary, i)

                else:
                    self.sess.run(train_step, {self.x: X, self.y_: Y_})

                # print(self.loss.eval({self.x: X, self.y_: Y_}))

            self.save_params()
            print('Total training loss: %f' % self.loss.eval({self.x: X, self.y_: Y_}))

    def outputs(self, X, Y_):
        # return
        # print(self.sess.run(self.c))
        for i in range(len(X)):
            print('Output on point #%i: %f' % (i, self.sess.run(self.y, {self.x: [X[i]], self.y_: [Y_[i]]})))
