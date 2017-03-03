from ConceptCell import concept_cell, input_dimension
from ReadLocationNames import locations
import tensorflow as tf
import hashlib

sess = tf.Session()
x, y_, pred, loss, step, init = concept_cell()
epoch_count = 1000

x_training = []
for location in locations:
    seq_hash = hashlib.sha256(location.encode('utf-8')).hexdigest()
    inputs = [int(seq_hash[8*i: 8*(i+1)], 16) for i in range(input_dimension)]
    x_training.append([inputs])

y_training = [[1]] * len(locations)

sess.run(init)
for i in range(epoch_count):
    print('epoch: %i' % i)
    sess.run(step, {x: x_training, y_: y_training})
