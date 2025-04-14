import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from experiments_runners.BaseExperimentRunner import BaseExperimentRunner

from config.TFTrasformer_settings import TFTransformerSettings

from models.TFTransformer.trainer import TFTransformerTrainer
from models.TFTransformer.utils import df_to_dataset


class TFTransformerExperimentRunner(BaseExperimentRunner):
    def __init__(self, train_data, test_data):
        super().__init__(train_data, test_data)
        self.config = TFTransformerSettings()

    def _run_single_experiment(self, class_type, aug, cw):
        exp_id = f"{class_type}_aug{aug}_cw{cw}"
        print(f"\n=== Running experiment: {exp_id} ===")

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

        # 4) Build & train model using LSTMTrainer
        num_classes = 1 if class_type == 'binary' else len(np.unique(y_train))
        class_weights = self.preprocessor.get_class_weights(y_train) if cw else None

        trainer = TFTransformerTrainer(experiment_id=exp_id)
        model, history = trainer.train(
            X_train, y_train,
            X_val, y_val,
            num_classes,
            class_type,
            class_weights
        )
        # 5) Now evaluate on the final test set
        X_test, y_test, _ = self.preprocessor.preprocess_validation_test(
            self.test_df, class_type
        )

        test_df_after_scaling = pd.DataFrame(np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1),
                                             columns=self.config.FEATURE_COLS + [class_type])

        test_dataset = df_to_dataset(test_df_after_scaling, class_type,
                                     shuffle=True, batch_size=1024)

        self._evaluate_and_save(exp_id, model, test_dataset, y_test, encoder)
        self._save_artifacts(exp_id, model, encoder)
        self._save_plots(exp_id, history, model, test_dataset, y_test, encoder)

        # except Exception as e:
        #     print(f"Experiment failed: {str(e)}")
        #     self._log_error(exp_id, str(e))

    def _evaluate_and_save(self, exp_id, model, X_test, y_test, encoder):
        y_pred = model.predict(X_test)['output']

        if model.output_layer.units == 1:  # Binary classification
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

    def _save_plots(self, exp_id, history, model, X_test, y_test, encoder):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['output_accuracy'], label='Train')
        plt.plot(history.history['val_output_accuracy'], label='Validation')
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
        if model.output_layer.units > 1:
            plt.figure(figsize=(15, 12))
            sns.heatmap(
                confusion_matrix(y_test, np.argmax(model.predict(X_test)['output'], axis=1)),
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
