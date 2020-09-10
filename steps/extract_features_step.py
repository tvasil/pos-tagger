from typing import Generator
from pipeline.pipeline import Step
from helpers import sent2features


class ExtractFeaturesForCRFFromTreebank(Step):
    """
    Step to extract features from a Treebank, which has a
    [([sentence], [tags])] structure
    """

    def run(self, batches: Generator):
        """
        Yields a generator for features, that feeds into the CRF model
        """
        for batch in batches:
            result = [[], []]
            for sentence in batch:
                result[0].append(sent2features(sentence[0]))
                result[1].append(sentence[1])

            yield result


class ExtractFeaturesForCRFFromList(Step):
    """
    Step to extract features from a batch of sentence lists, which has a
    ["token1 token2 ...", "token3 token4 ..."] structure. Tokens are assumed
    to be split by blank space (.split())
    """
    def run(self, batches: Generator):
        for sentence in batches:
            yield sent2features(sentence.split())
