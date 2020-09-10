import argparse
import pipeline

from steps.load_data_steps import LoadConlluDataStep
from steps.data_transform_steps import TransformConlluToTreebank
from steps.extract_features_step import ExtractFeaturesForCRFFromTreebank
from steps.evaluation_steps import CRFEvaluateStep

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a POS tagger')
    parser.add_argument('test_path',
                        type=str,
                        help='a .conllu train file for evaluating accuracy')
    args = parser.parse_args()

    eval_pipeline = pipeline.pipeline.Pipeline(
        steps=[LoadConlluDataStep([args.test_path], 10),
               TransformConlluToTreebank(),
               ExtractFeaturesForCRFFromTreebank(),
               CRFEvaluateStep('./crf_model.joblib')
               ]
    )
    eval_pipeline.run()
