from ReadLocationNames import locations
import tensorflow as tf

input_dimension = 8     # SHA-256


def concept_cell():
    # sess = tf.InteractiveSession()
    rbf_num = len(locations)
    alpha = 1.0
    beta = 1.0
    theta = 1.0

    x = tf.placeholder(tf.int64, shape=[None, 1, input_dimension])
    y_ = tf.placeholder(tf.int64, shape=[None, 1])
    c0 = tf.zeros([rbf_num, input_dimension])
    c = tf.Variable(tf.zeros([rbf_num, input_dimension]))  # TODO: initialization strategy
    r = tf.Variable(tf.ones([1, rbf_num]))  # TODO: initialization strategy & division-by-zero
    w = tf.ones([rbf_num, 1])  # Could be variables
    x_ = tf.tile(x, [1, rbf_num, 1])

    rbfs = tf.to_float(x_) - c          # Broadcasting feature. Cool!
    rbfs = tf.square(rbfs)
    rbfs = tf.reduce_sum(rbfs, 2)
    rbfs = tf.multiply(rbfs, tf.reciprocal(r))
    rbfs = -rbfs
    rbfs = tf.exp(rbfs)

    y = tf.tanh(tf.matmul(rbfs, w))          # No bias? Only one output?

    loss = tf.reduce_mean(tf.square(y - tf.to_float(y_))) \
        + alpha * tf.reduce_sum(beta * tf.sigmoid(tf.square(c0 - c))) \
        + theta * tf.reduce_sum(tf.square(r))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    init = tf.global_variables_initializer()

    return x, y_, y, loss, train_step, init

# sess.run(tf.global_variables_initializer())
