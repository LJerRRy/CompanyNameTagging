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

    x = tf.placeholder(tf.float32, shape=[1, None])  # TODO: type?
    y_ = tf.placeholder(tf.float32, shape=[1, None])
    c0 = tf.zeros([rbf_num, 1])
    c = tf.Variable(tf.zeros([rbf_num, 1]))  # TODO: initialization strategy
    r = tf.Variable(tf.ones([rbf_num, 1]))  # TODO: initialization strategy & division-by-zero
    W = tf.ones([1, rbf_num])  # Could be variables
    x = tf.tile(x, [rbf_num, 1])

    rbfs = (x - c) / r          # Broadcasting feature. Cool!
    rbfs = tf.square(rbfs)
    rbfs = -rbfs
    rbfs = tf.exp(rbfs)

    y = tf.tanh(tf.matmul(W, rbfs))          # No bias? Only one output?

    loss = tf.reduce_mean(tf.square(y - y_)) \
           + alpha * tf.reduce_sum(beta * tf.sigmoid(tf.square(c0 - c))) \
           + theta * tf.reduce_sum(tf.square(r))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    return y, loss, train_step

# sess.run(tf.global_variables_initializer())
