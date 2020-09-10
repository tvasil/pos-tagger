import time
from sklearn_crfsuite import CRF
from os import path
from typing import Generator

from pipeline.pipeline import Step


class CRFTrainStep(Step):
    """
    Step to train a CRF model iteratively by updating the matrices.
    """
    def __init__(self, model_file_path):
        # load the model if it exists
        self.model_file_path = path.abspath(path.expanduser(model_file_path))
        self.model = CRF(
                algorithm='l2sgd',
                c2=0.1,
                max_iterations=1000,
                all_possible_transitions=True,
                model_filename=self.model_file_path)

    def run(self, batches: Generator):
        """
        Runs the step
        """
        st = time.time()

        x = []
        y = []
        for batch in batches:
            b = list(batch)
            x.extend(b[0])
            y.extend(b[1])

        self.model = self.model.fit(x, y)

        et = time.time()
        print(f"The CRF model finished training in {round(et-st, 2)} seconds.")
