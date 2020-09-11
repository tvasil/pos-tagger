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

        accuracy = self.model.score(x, y)
        y_pred = self.model.predict(x)
        f1_score = metrics.flat_f1_score(y, y_pred)
        accuracy_sentence = metrics.sequence_accuracy_score(y, y_pred)
        classification_report = metrics.flat_classification_report(
            y,
            y_pred,
            labels=self.model.classes_)
        print("*"*80)
        print("MODEL EVALUATION")
        print("*"*80)
        print("Token-wise accuracy score on Test Data:")
        print(round(accuracy, 3))
        print("F1 score on Test Data:")
        print(round(f1_score, 3))
        print("Sequence accurancy score (% of sentences scored 100% correctly):")
        print(round(accuracy_sentence, 3))
        print("Class-wise classification report:")
        print(classification_report)
        et = time.time()
        print(f"Evaluation finished in {round(et-st, 2)} seconds.")
