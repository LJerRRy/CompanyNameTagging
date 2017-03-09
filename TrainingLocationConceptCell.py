from ConceptCell import ConceptCell
from ReadLocationNames import locations
import tensorflow as tf
import hashlib

model_path = './model/location_model'
location_cell = ConceptCell('location_concept_cell')

x_training = []
for location in locations:
    seq_hash = hashlib.sha256(location.encode('utf-8')).hexdigest()
    inputs = [int(seq_hash[8*i: 8*(i+1)], 16) for i in range(location_cell.input_dimension)]
    x_training.append([inputs])

y_training = [[1]] * len(locations)

# location_cell.train(x_training, y_training)

# location_cell.train(x_training, y_training)
location_cell.print_r()
