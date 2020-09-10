from typing import Generator
from pipeline.pipeline import Step


class TransformConlluToTreebank(Step):

    def run(self, batches: Generator):
        """
        Kwargs must contain the key "batches" with has the batches read
        from the conllu files.
        """
        for batch in batches:
            result = []
            for sentence in batch:
                tokens = []
                tags = []
                for token in sentence:
                    tokens.append(token['form'])
                    tags.append(token['upostag'])
                result.append((tokens, tags))
            yield result
