from ReadLocationNames import locations
import tensorflow as tf

rbf_num = len(locations)
alpha = 1.0
beta = 1.0
theta = 1.0

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[1, None])     # TODO: type?
y_ = tf.placeholder(tf.float32, shape=[1, None])
c0 = tf.zeros([rbf_num, 1])
c = tf.Variable(tf.zeros([rbf_num, 1]))     # TODO: initialization strategy
r = tf.Variable(tf.ones([rbf_num, 1]))      # TODO: initialization strategy & division-by-zero
W = tf.ones([1, rbf_num])                   # Could be variables
x = tf.tile(x, [rbf_num, 1])


def concept_cell(x, c, r, W):
    rbfs = tf.slice(x, [0, 0], [-1, 1]) - c
    rbfs = tf.multiply(rbfs, tf.reciprocal(r))
    for i in range(1, sess.run(tf.shape(x)[1])):
        rbf_current_case = tf.slice(x, [0, i], [-1, 1]) - c
        rbf_current_case = tf.multiply(rbf_current_case, tf.reciprocal(r))
        rbfs = tf.concat(1, [rbfs, rbf_current_case])

    rbfs = tf.multiply(rbfs, rbfs)
    rbfs = -rbfs
    rbfs = tf.exp(rbfs)

    y = tf.tanh(tf.matmul(W, rbfs))          # No bias? Only one output?
    return y


pred = concept_cell(x, c, r, W)

loss = tf.reduce_mean(tf.square(pred - y_)) \
       + alpha * tf.reduce_sum(beta * tf.sigmoid(tf.square(c0 - c)))\
       + theta * tf.reduce_sum(tf.square(r))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess.run(tf.global_variables_initializer())
