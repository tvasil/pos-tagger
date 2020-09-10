import time
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics

from typing import Generator

from os import path
from pipeline.pipeline import Step


class CRFEvaluateStep(Step):
    """
    Step to evaluate testing data against a CRF model,
    stored on file
    """
    def __init__(self, model_file_path):
        self.model_file_path = path.abspath(path.expanduser(model_file_path))
        self.model = CRF(
                algorithm='l2sgd',
                c2=0.1,
                max_iterations=1000,
                all_possible_transitions=True,
                model_filename=self.model_file_path)

    def run(self, batches: Generator) -> None:
        """
        Runs the CRF model, storing to pickle in the end
        """
        st = time.time()

        x = []
        y = []

        # For prediction, CRF does not implement batching, so we pass a list
        for batch in batches:
            b = list(batch)
            x.extend(b[0])
            y.extend(b[1])

        score = self.model.score(x, y)
        y_pred = self.model.predict(x)
        score_sentence = metrics.sequence_accuracy_score(y, y_pred)
        classification_report = metrics.flat_classification_report(
            y,
            y_pred,
            labels=self.model.classes_)
        print("*"*80)
        print("F1 score on Test Data")
        print(round(score, 3))
        print("Sequence accurancy score (% of sentences scored 100% correctly):")
        print(round(score_sentence, 3))
        print("Class-wise classification report:")
        print(classification_report)
        et = time.time()
        print(f"The model finished evaluating in {round(et-st, 2)} seconds.")
