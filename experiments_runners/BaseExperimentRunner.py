import pandas as pd

from config.general_settings import GeneralSettings
from data.code.preprocessor import DataPreprocessor
import itertools

from joblib import dump


class BaseExperimentRunner:
    def __init__(self, train_data, test_data):
        self.config = GeneralSettings()
        self.train_df = train_data
        self.test_df = test_data
        self.preprocessor = DataPreprocessor()
        self.results = []

    def run_experiments(self):
        self._run_combination_experiments()
        self._run_window_experiments()
        self._save_results()

    def _run_combination_experiments(self):
        for combo in itertools.product(
                self.config.CLASS_TYPES,
                self.config.AUGMENTATIONS,
                self.config.CLASS_WEIGHTS
        ):
            self._run_single_experiment(*combo)

    def _run_window_experiments(self):
        pass  # To be optionally overridden

    def _log_error(self, exp_id, error):
        self.results.append({
            'experiment': exp_id,
            'error': error,
            'status': 'failed'
        })

    def _save_artifacts(self, exp_id, model, encoder):
        model.save(self.config.MODELS_PATH / f"{exp_id}_model.h5")
        dump(self.preprocessor.scaler, self.config.SCALERS_PATH / f"{exp_id}_scaler.joblib")
        if encoder:
            dump(encoder, self.config.ENCODERS_PATH / f"{exp_id}_encoder.joblib")

    def _save_results(self):
        pd.DataFrame(self.results).to_csv(self.config.RESULTS_PATH / 'experiment_results.csv', index=False)

    def _run_single_experiment(self, class_type, aug, cw):
        raise NotImplementedError("Must be implemented by subclass")

    def _evaluate_and_save(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by subclass")

    def _save_plots(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented by subclass")
