from ConceptCell import init, concept_cell
from ReadLocationNames import locations
import tensorflow as tf

sess = tf.Session()
sess.run(init)
pred, loss, step = concept_cell()


