from ReadLocationNames import locations
import tensorflow as tf

input_dimension = 8     # SHA-256
init = tf.global_variables_initializer()


def concept_cell():
    # One-Dimensional
    # sess = tf.InteractiveSession()
    rbf_num = len(locations)
    alpha = 1.0
    beta = 1.0
    theta = 1.0

    x = tf.placeholder(tf.int32, shape=[None, 1, input_dimension])
    y_ = tf.placeholder(tf.int32, shape=[None, 1])
    c0 = tf.zeros([rbf_num, input_dimension])
    c = tf.Variable(tf.zeros([rbf_num, input_dimension]))  # TODO: initialization strategy
    r = tf.Variable(tf.ones([1, rbf_num]))  # TODO: initialization strategy & division-by-zero
    w = tf.ones([rbf_num, 1])  # Could be variables
    x = tf.tile(x, [1, rbf_num, 1])

    rbfs = (x - c)          # Broadcasting feature. Cool!
    rbfs = tf.square(rbfs)
    rbfs = tf.reduce_sum(rbfs, 2)
    rbfs = tf.multiply(rbfs, tf.reciprocal(r))
    rbfs = -rbfs
    rbfs = tf.exp(rbfs)

    y = tf.tanh(tf.matmul(rbfs, w))          # No bias? Only one output?

    loss = tf.reduce_mean(tf.square(y - y_)) \
        + alpha * tf.reduce_sum(beta * tf.sigmoid(tf.square(c0 - c))) \
        + theta * tf.reduce_sum(tf.square(r))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    return y, loss, train_step

# sess.run(tf.global_variables_initializer())
