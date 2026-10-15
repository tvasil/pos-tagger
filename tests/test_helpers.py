import unittest

from helpers import extract_features, tokenize
from steps.data_transform_steps import TransformConlluToTreebank


class TokenizerTests(unittest.TestCase):
    def test_separates_punctuation_and_keeps_apostrophes(self):
        self.assertEqual(
            tokenize("I can't fish."),
            ["I", "can't", "fish", "."],
        )


class FeatureTests(unittest.TestCase):
    def test_context_features_for_can(self):
        features = extract_features(["I", "can", "fish", "."], 1)
        self.assertEqual(features["word.lower"], "can")
        self.assertEqual(features["structure"], "xxx")
        self.assertEqual(features["prev.word"], "I")
        self.assertEqual(features["next.word"], "fish")


class ConlluTransformTests(unittest.TestCase):
    def test_ignores_multiword_rows_without_upos(self):
        sentence = [
            {"id": (1, "-", 2), "form": "can't", "upostag": "_"},
            {"id": 1, "form": "ca", "upostag": "AUX"},
            {"id": 2, "form": "n't", "upostag": "PART"},
        ]
        output = list(TransformConlluToTreebank().run([[sentence]]))
        self.assertEqual(output, [[(["ca", "n't"], ["AUX", "PART"])]])


if __name__ == "__main__":
    unittest.main()
