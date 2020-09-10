from sklearn_crfsuite import CRF

from typing import Generator

from os import path
from pipeline.pipeline import Step


class CRFPredictStep(Step):
    def __init__(self, model_file_path):
        self.model_file_path = path.abspath(path.expanduser(model_file_path))
        self.model = CRF(
                algorithm='l2sgd',
                c2=0.1,
                max_iterations=1000,
                all_possible_transitions=True,
                model_filename=self.model_file_path)

    def run(self, batches: Generator):
        features = list(batches)
        pred = self.model.predict(features)
        for index, feature in enumerate(features):
            print(' '.join(map(lambda x: x['word'], feature)), end='')
            print(' => ', end='')
            print(pred[index])
