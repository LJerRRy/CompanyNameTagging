import hashlib
import numpy as np

from ConceptCell import ConceptCell
from ReadCompanyTypes import company_types


x_training = []
for company_type in company_types:
    seq_hash = hashlib.sha256(company_type.encode('utf-8')).hexdigest()
    inputs = [int(seq_hash[8 * i: 8 * (i + 1)], 16) for i in range(ConceptCell.input_dimension)]
    x_training.append([inputs])

y_training = [[1]] * len(company_types)
company_type_cell = ConceptCell('company_type_concept_cell', x_training, y_training)


def train_the_model():
    # company_type_cell.train(x_training, y_training)
    company_type_cell.outputs(x_training, y_training)


def test_the_model():
    x_testing = np.array(x_training) + 1
    y_testing = [[0]] * len(company_types)
    company_type_cell.outputs(x_testing, y_testing)


train_the_model()
# test_the_model()
