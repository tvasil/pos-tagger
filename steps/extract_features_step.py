from typing import Generator
from pipeline.pipeline import Step
from helpers import sent2features


class ExtractFeaturesForCRFFromTreebank(Step):

    def run(self, batches: Generator):
        for batch in batches:
            result = [[], []]
            for sentence in batch:
                result[0].append(sent2features(sentence[0]))
                result[1].append(sentence[1])

            yield result


class ExtractFeaturesForCRFFromList(Step):

    def run(self, batches: Generator):
        for sentence in batches:
            yield sent2features(sentence.split())
