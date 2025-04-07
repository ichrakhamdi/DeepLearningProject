import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from joblib import dump
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config.settings import LSTMSettings
from data.code.preprocessor import DataPreprocessor
from models.lstm.trainer import LSTMTrainer

train_df = pd.read_csv("data/concatenated/train.csv")
test_df = pd.read_csv("data/concatenated/test.csv")

class LSTMExperimentRunner:
    def __init__(self, train_data, test_data):
        self.config = LSTMSettings()
        self.preprocessor = DataPreprocessor()
        self.results = []
        self.train_df = train_data
        self.test_df = test_data

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
        pass

    def _run_single_experiment(self, class_type, aug, cw):
        exp_id = f"{class_type}_aug{aug}_cw{cw}"
        print(f"\n=== Running experiment: {exp_id} ===")

        try:
            # 1) Split the "train_df" into train vs. validation

            train_df_split, val_df_split = train_test_split(
                self.train_df,
                test_size=0.2,
                random_state=42
            )

            # 2) Preprocess the TRAIN portion (with augmentation if requested)
            X_train, y_train, encoder = self.preprocessor.preprocess_train(
                train_df_split, class_type, augmentation=aug
            )

            # 3) Preprocess the VALIDATION portion (no augmentation)
            X_val, y_val, _ = self.preprocessor.preprocess_validation_test(
                val_df_split, class_type
            )

            # 4) Reshape for LSTM input (samples, timesteps, features)
            X_train = X_train.reshape(-1, 1, len(self.config.FEATURE_COLS))
            X_val = X_val.reshape(-1, 1, len(self.config.FEATURE_COLS))

            # 5) Build & train model using LSTMTrainer 
            num_classes = 1 if class_type == 'binary' else len(np.unique(y_train))
            class_weights = self.preprocessor.get_class_weights(y_train) if cw else None

            trainer = LSTMTrainer(experiment_id=exp_id)
            model, history = trainer.train(
                X_train, y_train,
                X_val, y_val,
                num_classes,
                class_weights
            )

            # 6) Now evaluate on the final test set
            X_test, y_test, _ = self.preprocessor.preprocess_validation_test(
                self.test_df, class_type
            )
            X_test = X_test.reshape(-1, 1, len(self.config.FEATURE_COLS))

            self._evaluate_and_save(exp_id, model, X_test, y_test, encoder)
            self._save_artifacts(exp_id, model, encoder)
            self._save_plots(exp_id, history, model, X_test, y_test, encoder)

        except Exception as e:
            print(f"Experiment failed: {str(e)}")
            self._log_error(exp_id, str(e))

    def _evaluate_and_save(self, exp_id, model, X_test, y_test, encoder):
        y_pred = model.predict(X_test)

        if model.output_shape[-1] == 1:  # Binary classification
            y_pred_class = (y_pred > 0.5).astype(int)
            class_names = ['Benign', 'Attack']
        else:
            y_pred_class = np.argmax(y_pred, axis=1)
            class_names = encoder.classes_ if encoder is not None else [
                str(i) for i in range(model.output_shape[-1])
            ]

        # Classification report & confusion matrix
        report = classification_report(y_test, y_pred_class, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_test, y_pred_class)

        pd.DataFrame(report).transpose().to_csv(self.config.RESULTS_PATH / f"{exp_id}_report.csv")
        pd.DataFrame(cm).to_csv(self.config.RESULTS_PATH / f"{exp_id}_cm.csv")

        self.results.append({
            'experiment': exp_id,
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score']
        })

    def _save_artifacts(self, exp_id, model, encoder):
        model.save(self.config.MODELS_PATH / f"{exp_id}_model.h5")
        dump(self.preprocessor.scaler, self.config.SCALERS_PATH / f"{exp_id}_scaler.joblib")
        if encoder:
            dump(encoder, self.config.ENCODERS_PATH / f"{exp_id}_encoder.joblib")

    def _save_plots(self, exp_id, history, model, X_test, y_test, encoder):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title(f'{exp_id} Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title(f'{exp_id} Loss')
        plt.legend()
        plt.savefig(self.config.PLOTS_PATH / f"{exp_id}_curves.png")
        plt.close()

        # Confusion matrix if multi-class
        if model.output_shape[-1] > 1:
            plt.figure(figsize=(15, 12))
            sns.heatmap(
                confusion_matrix(y_test, np.argmax(model.predict(X_test), axis=1)),
                annot=True, fmt='d',
                xticklabels=encoder.classes_ if encoder is not None else None,
                yticklabels=encoder.classes_ if encoder is not None else None
            )
            plt.title(f'{exp_id} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(self.config.PLOTS_PATH / f"{exp_id}_cm.png")
            plt.close()

    def _log_error(self, exp_id, error):
        self.results.append({
            'experiment': exp_id,
            'error': error,
            'status': 'failed'
        })

    def _save_results(self):
        pd.DataFrame(self.results).to_csv(self.config.RESULTS_PATH / 'experiment_results.csv', index=False)

if __name__ == "__main__":
    runner = LSTMExperimentRunner(train_df, test_df)
    runner.run_experiments()
    print("=== All experiments completed successfully ===")
