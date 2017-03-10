import hashlib
import numpy as np

from ConceptCell import ConceptCell
from ReadCompanyTypes import read_company_types


class CompanyTypesConceptCell(ConceptCell):
    x_training = None
    y_training = None
    x_testing = None
    y_testing = None

    def __init__(self):
        self.x_training = []
        company_types = read_company_types()
        for company_type in company_types:
            seq_hash = hashlib.sha256(company_type.encode('utf-8')).hexdigest()  # Have to encode string before hash
            inputs = [int(seq_hash[8 * i: 8 * (i + 1)], 16) for i in range(ConceptCell.input_dimension)]
            self.x_training.append([inputs])

        self.y_training = [[1]] * len(company_types)

        self.x_testing = np.array(self.x_training) + 1
        self.y_testing = [[0]] * len(company_types)

        super(CompanyTypesConceptCell, self).__init__('location_concept_cell', self.x_training, self.y_training)

    def train_model(self):
        super(CompanyTypesConceptCell, self).train(self.x_training, self.y_training)

    def test_model(self):
        super(CompanyTypesConceptCell, self).outputs(self.x_testing, self.y_testing)

    def activation(self, X):
        return super(CompanyTypesConceptCell, self).activation(X)

    def outputs_for_training_set(self):
        super(CompanyTypesConceptCell, self).outputs(self.x_training, self.y_training)
