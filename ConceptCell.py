from ReadLocationNames import locations
import tensorflow as tf

rbf_num = len(locations)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[1, None])
y_ = tf.placeholder(tf.float32, shape=[1, None])
c = tf.Variable(tf.zeros([rbf_num, 1]))     # TODO: initialization strategy
r = tf.Variable(tf.ones([rbf_num, 1]))      # TODO: initialization strategy & division-by-zero

x = tf.tile(x, [rbf_num, 1])

RBFs = tf.slice(x, [0, 0], [-1, 1]) - c
for i in range(1, tf.shape(x)[1]):
    RBFs = tf.concat(1, [RBFs, tf.slice(x, [0, i], [-1, 1]) - c])

RBFs = tf.multiply(RBFs, RBFs)
