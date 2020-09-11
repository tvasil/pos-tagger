import argparse
import pipeline

from steps.load_data_steps import LoadTxtDataStep
from steps.extract_features_step import ExtractFeaturesForCRFFromList
from steps.predict_steps import CRFPredictStep

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get new predictions')
    parser.add_argument('test_path',
                        type=str,
                        help='txt file to get predictions on')
    args = parser.parse_args()

    eval_pipeline = pipeline.pipeline.Pipeline(
        steps=[LoadTxtDataStep(args.test_path),
               ExtractFeaturesForCRFFromList(),
               CRFPredictStep('./crf_model.joblib')
               ]
    )
    eval_pipeline.run()
