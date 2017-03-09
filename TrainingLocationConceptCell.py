import hashlib

from ConceptCell import ConceptCell
from ReadLocationNames import locations


def train_the_model():
    x_training = []
    for location in locations:
        seq_hash = hashlib.sha256(location.encode('utf-8')).hexdigest()
        inputs = [int(seq_hash[8*i: 8*(i+1)], 16) for i in range(ConceptCell.input_dimension)]
        x_training.append([inputs])

    y_training = [[1]] * len(locations)

    model_path = './model/location_model'
    location_cell = ConceptCell('location_concept_cell', x_training, y_training)

    # location_cell.train(x_training, y_training)
    # location_cell.print_r()
    location_cell.outputs(x_training, y_training)

def test_the_model():
    # All nil's. Yeah!
    x_training = []
    for location in locations:
        seq_hash = hashlib.sha256(location.encode('utf-8')).hexdigest()
        inputs = [int(seq_hash[8 * i: 8 * (i + 1)], 16)+1 for i in range(ConceptCell.input_dimension)]
        x_training.append([inputs])

    y_training = [[1]] * len(locations)

    location_cell = ConceptCell('location_concept_cell', x_training, y_training)
    location_cell.outputs(x_training, y_training)

# train_the_model()
# test_the_model()
