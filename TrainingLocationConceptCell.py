import hashlib
import numpy as np

from ConceptCell import ConceptCell
from ReadLocationNames import read_location_names


locations = read_location_names()

x_training = []
for location in locations:
    seq_hash = hashlib.sha256(location.encode('utf-8')).hexdigest()  # Have to encode string before hash
    inputs = [int(seq_hash[8 * i: 8 * (i + 1)], 16) for i in range(ConceptCell.input_dimension)]
    x_training.append([inputs])

y_training = [[1]] * len(locations)

location_cell = ConceptCell('location_concept_cell', x_training, y_training)


# TODO: class inheritance & template
# TODO: check whether model has been trained
def train_the_model():
    location_cell.train(x_training, y_training)
    # location_cell.print_r()
    location_cell.outputs(x_training, y_training)


def get_location_concept_cell():
    location_cell.restore_params()
    return location_cell


def test_the_model():
    # All nil's. Yeah!
    # x_training = []
    # for location in locations:
    #     seq_hash = hashlib.sha256(location.encode('utf-8')).hexdigest()
    #     inputs = [int(seq_hash[8 * i: 8 * (i + 1)], 16)+1 for i in range(ConceptCell.input_dimension)]
    #     x_training.append([inputs])

    x_testing = np.array(x_training) + 1
    y_testing = [[0]] * len(locations)
    location_cell.outputs(x_testing, y_testing)

# train_the_model()
# test_the_model()
