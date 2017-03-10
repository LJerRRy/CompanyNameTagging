import hashlib
import numpy as np

from ConceptCell import ConceptCell
from ReadLocationNames import read_location_names


class LocationConceptCell(ConceptCell):
    x_training = None
    y_training = None
    x_testing = None
    y_testing = None

    def __init__(self):
        self.x_training = []
        locations = read_location_names()
        for location in locations:
            seq_hash = hashlib.sha256(location.encode('utf-8')).hexdigest()  # Have to encode string before hash
            inputs = [int(seq_hash[8 * i: 8 * (i + 1)], 16) for i in range(ConceptCell.input_dimension)]
            self.x_training.append([inputs])

        self.y_training = [[1]] * len(locations)

        self.x_testing = np.array(self.x_training) + 1
        self.y_testing = [[0]] * len(locations)

        super(LocationConceptCell, self).__init__('location_concept_cell', self.x_training, self.y_training)

    def train_model(self):
        super(LocationConceptCell, self).train(self.x_training, self.y_training)

    def test_model(self):
        super(LocationConceptCell, self).outputs(self.x_testing, self.y_testing)

    def activation(self, X):
        return super(LocationConceptCell, self).activation(X)

    def outputs_for_training_set(self):
        super(LocationConceptCell, self).outputs(self.x_training, self.y_training)
