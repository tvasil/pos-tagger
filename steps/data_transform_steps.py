from typing import Generator
from pipeline.pipeline import Step


class TransformConlluToTreebank(Step):
    """Step to tranform files read from a Conllu structure
    to a treebank, which consists of [([sentence], [tags])]"""

    def run(self, batches: Generator) -> Generator:
        """
        Runs the tranformation from .conllu files to a treebank
        which has the structure of (([sentence], [tags]))
        """
        for batch in batches:
            result = []
            for sentence in batch:
                tokens = []
                tags = []
                for token in sentence:
                    # CoNLL-U multiword-token rows use range IDs and do not
                    # carry UPOS labels. Only syntactic word rows belong in
                    # the CRF sequence.
                    if not isinstance(token['id'], int):
                        continue
                    if token['upostag'] in (None, '_'):
                        continue
                    tokens.append(token['form'])
                    tags.append(token['upostag'])
                result.append((tokens, tags))
            yield result
