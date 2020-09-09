import random
import time
from typing import Generator

from joblib import dump, load
import numpy as np
import pandas as pd
from scipy import stats
from sklearn_crfsuite import CRF, metrics, scorers
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

from helpers import *


class ConditionalRandomFieldsEstimator():
    """Wrapper class for sklearn_crfsuite.CRF. Ensures model persistence by
    saving the model via joblib. Also conditionally implements cross validation score"""
    def __init__(self,
                 algorithm:str = 'l2sgd',
                 model_filename:str = 'crf_model.joblib'):
        self.algorithm = algorithm
        self.model_filename = model_filename
        self.model = None
        self.preds = []


    def fit(self,
            X_train: Generator,
            y_train: list,
            X_test: Generator,
            y_test: list) -> None:
        st = time.time()
        if not self.model:
            self.model = CRF(
                algorithm=self.algorithm,
                c2=0.1,
                max_iterations=1000,
                all_possible_transitions=True)
        else:
            self.model = joblib.load(self.model_filename)
        self.model.fit(
            X_train,
            y_train,
            X_test,
            y_test)

        joblib.dump(self.model,
                    self.model_filename)
        self.evaluate_training_accuracy(
            X_train,
            y_train,
            X_test,
            y_test
        )
        et = time.time()
        print(f"The model finished training in {round(et-st, 2)} seconds.")


    def predict(self, X_test: Generator):
        self.preds = self.model.predict(X_test)


    def evaluate_training_accuracy(self,
                                   X_train: Generator,
                                   y_train: list,
                                   X_test: Generator,
                                   y_test: list) -> None:
        print("F1 score on Train Data")
        print(self.model.score(X_train, y_train))
        print("F1 score on Dev/Test Data")
        print(self.model.score(X_test, y_test))


    def cross_validate(self,
                       X_train:list,
                       y_train:list):
        f1_scorer = make_scorer(metrics.flat_f1_score, average='macro')
        scores = cross_validate(self.model, X_train, y_train, scoring=f1_scorer, cv=5)
        print(scores)


    def print_classification_report(self,
                                    y_test):
        print("Class wise score:")
        print(metrics.flat_classification_report(
                        y_test,
                        self.preds,
                        labels=self.model.classes_,
                        digits=3
                        )
             )

class TrainingPipeline():
    def read_conllu_files_to_list(filename:str) -> list:
        files = read_file(filename)
        return files_to_treebank(files)


    def extract_features(treebank:list) -> tuple:
        X = [sent2features(s[0]) for s in treebank]
        y = [s[1] for s in treebank]
        return X, y


    def train_model(X_train, y_train, )
