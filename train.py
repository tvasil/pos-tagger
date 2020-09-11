import argparse

import pipeline
from steps.load_data_steps import LoadConlluDataStep
from steps.data_transform_steps import TransformConlluToTreebank
from steps.extract_features_step import ExtractFeaturesForCRFFromTreebank
from steps.training_steps import CRFTrainStep

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a POS tagger')
    parser.add_argument('train_path',
                        type=str,
                        help='a .conllu train file for training')
    parser.add_argument('dev_path',
                        type=str,
                        help='a .conllu devn file for training')
    args = parser.parse_args()

    print("Training POS tagger with Conditional Random Fields")
    print("*"*80)

    training_pipeline = pipeline.pipeline.Pipeline(
        steps=[LoadConlluDataStep([args.train_path, args.dev_path], 10),
               TransformConlluToTreebank(),
               ExtractFeaturesForCRFFromTreebank(),
               CRFTrainStep('./crf_model.joblib')
               ]
    )
    training_pipeline.run()
