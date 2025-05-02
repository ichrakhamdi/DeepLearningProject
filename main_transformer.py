import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from joblib import dump, load
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
import shutil

# Use Transformer settings and trainer
from config.transformer_settings import TransformerSettings
from data.code.preprocessor import DataPreprocessor
from models.transformer.trainer import TransformerTrainer

# Load data once
try:
    train_df = pd.read_csv("data/concatenated/train.csv")
    test_df = pd.read_csv("data/concatenated/test.csv")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure 'data/concatenated/train.csv' and 'test.csv' exist.")
    print("You might need to run the data preparation script first.")
    exit() # Exit if data is not found


class TransformerExperimentRunner:
    def __init__(self, train_data, test_data):
        self.config = TransformerSettings() # Use TransformerSettings
        self.preprocessor = DataPreprocessor()
        self.results = []
        self.train_df = train_data
        self.test_df = test_data
        self.best_val_loss = float('inf')
        self.best_model_exp_id = None
        self.best_model_path = None
        self.best_scaler_path = None
        self.best_encoder_path = None

    def run_experiments(self):
        self._run_combination_experiments()
        self._save_overall_best_model()
        self._save_results()

    def _run_combination_experiments(self):
        for combo in itertools.product(
            self.config.CLASS_TYPES,
            self.config.AUGMENTATIONS,
            self.config.CLASS_WEIGHTS
        ):
            self._run_single_experiment(*combo)

    def _run_single_experiment(self, class_type, aug, cw):
        preprocessor = DataPreprocessor()

        exp_id = f"transformer_{class_type}_aug{aug}_cw{cw}_win{self.config.WINDOW_SIZE}"
        print(f"\n=== Running Transformer experiment: {exp_id} ===")

        try:
            train_df_split, val_df_split = train_test_split(
                self.train_df,
                test_size=0.2,
                random_state=42,
                stratify=self.train_df[self.config.CLASS_MAP[class_type]] if class_type in self.config.CLASS_MAP else None
            )
            print(f"[Data Split] Train: {len(train_df_split)}, Validation: {len(val_df_split)}, Test: {len(self.test_df)}")

            X_train_processed, y_train_processed, encoder = preprocessor.preprocess_train(
                train_df_split, class_type, augmentation=aug
            )
            print(f"[Preprocessing Train] Shape after scaling/encoding (and augmentation={aug}): X={X_train_processed.shape}, y={y_train_processed.shape}")

            X_val_processed, y_val_processed, _ = preprocessor.preprocess_validation_test(
                val_df_split, class_type
            )
            print(f"[Preprocessing Val] Shape after scaling/encoding: X={X_val_processed.shape}, y={y_val_processed.shape}")

            X_test_processed, y_test_processed, _ = preprocessor.preprocess_validation_test(
                self.test_df, class_type
            )
            print(f"[Preprocessing Test] Shape after scaling/encoding: X={X_test_processed.shape}, y={y_test_processed.shape}")

            window_size = self.config.WINDOW_SIZE
            print(f"[Sequence Creation] Using window size: {window_size}")

            if len(X_train_processed) < window_size or len(X_val_processed) < window_size or len(X_test_processed) < window_size:
                 warnings.warn(f"Skipping experiment {exp_id}: Not enough data points ({len(X_train_processed)} train, {len(X_val_processed)} val, {len(X_test_processed)} test) for window size {window_size}.")
                 self._log_error(exp_id, f"Insufficient data for window size {window_size}")
                 return

            X_train_seq, y_train_seq = preprocessor.create_sequences(X_train_processed, y_train_processed, window_size)
            X_val_seq, y_val_seq = preprocessor.create_sequences(X_val_processed, y_val_processed, window_size)
            X_test_seq, y_test_seq = preprocessor.create_sequences(X_test_processed, y_test_processed, window_size)
            print(f"[Sequence Shapes] Train: X={X_train_seq.shape}, y={y_train_seq.shape} | Val: X={X_val_seq.shape}, y={y_val_seq.shape} | Test: X={X_test_seq.shape}, y={y_test_seq.shape}")

            if X_train_seq.shape[0] == 0 or X_val_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
                warnings.warn(f"Skipping experiment {exp_id}: Empty sequences after applying window size {window_size}.")
                self._log_error(exp_id, f"Empty sequences for window size {window_size}")
                return

            num_classes = 1 if class_type == 'binary' else len(np.unique(y_train_seq))
            print(f"[Model Params] num_classes: {num_classes} (derived from y_train_seq)")
            class_weights_dict = preprocessor.get_class_weights(y_train_seq) if cw else None
            if class_weights_dict:
                print(f"[Model Params] Applying class weights: {class_weights_dict}")

            trainer = TransformerTrainer(experiment_id=exp_id)
            print(f"[Training] Starting training for {self.config.EPOCHS} epochs (Batch size: {self.config.BATCH_SIZE})...")
            model, history = trainer.train(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                num_classes,
                class_weights_dict
            )
            print("[Training] Training finished.")

            try:
                current_min_val_loss = min(history.history['val_loss'])
                print(f"[Tracking] Min validation loss for this run: {current_min_val_loss:.4f}")
                if current_min_val_loss < self.best_val_loss:
                    self.best_val_loss = current_min_val_loss
                    self.best_model_exp_id = exp_id
                    self.best_model_path = self.config.MODELS_PATH / f'best_transformer_model_{exp_id}.h5'
                    self.best_scaler_path = self.config.SCALERS_PATH / f"{exp_id}_scaler.joblib"
                    self.best_encoder_path = self.config.ENCODERS_PATH / f"{exp_id}_encoder.joblib" if encoder else None
                    print(f"*** New best model found! exp_id: {exp_id}, val_loss: {self.best_val_loss:.4f} ***")
            except KeyError:
                warnings.warn(f"Could not track best model for {exp_id}: 'val_loss' not found in history.")
            except ValueError:
                 warnings.warn(f"Could not track best model for {exp_id}: 'val_loss' history is empty.")

            print("[Evaluation] Evaluating model on test set...")
            self._evaluate_and_save(exp_id, model, X_test_seq, y_test_seq, encoder)

            print("[Saving] Saving artifacts and plots for this run...")
            self._save_run_artifacts(exp_id, preprocessor.scaler, encoder)
            self._save_plots(exp_id, history, model, X_test_seq, y_test_seq, encoder)
            print(f"=== Experiment {exp_id} completed successfully ===")

        except Exception as e:
            print(f"!!! Experiment {exp_id} failed: {str(e)} !!!")
            import traceback
            traceback.print_exc()
            self._log_error(exp_id, str(e))

    def _evaluate_and_save(self, exp_id, model, X_test_seq, y_test_seq, encoder):
        eval_results = model.evaluate(X_test_seq, y_test_seq, verbose=0)
        loss = eval_results[0]
        acc = eval_results[1]
        print(f"[Evaluation] Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
        y_pred = model.predict(X_test_seq)

        if model.output_shape[-1] == 1:  # Binary classification
            y_pred_class = (y_pred > 0.5).astype(int).flatten()
            y_test_class = y_test_seq.flatten() if y_test_seq.ndim > 1 else y_test_seq
            class_names = ['Benign', 'Attack']
            num_classes_eval = 2
        else: # Multi-class
            y_pred_class = np.argmax(y_pred, axis=1)
            y_test_class = y_test_seq
            num_classes_eval = model.output_shape[-1]
            if encoder is not None and hasattr(encoder, 'classes_'):
                class_names = encoder.classes_
            else:
                 class_names = [str(i) for i in range(num_classes_eval)]
                 warnings.warn("Label encoder not found or missing 'classes_' attribute. Using generic class names.")
            print(f"[Evaluation] Multi-class detected. Target classes: {class_names}")

        if y_test_class.shape != y_pred_class.shape:
             warnings.warn(f"Shape mismatch for evaluation: y_test {y_test_class.shape}, y_pred {y_pred_class.shape}. Skipping metrics.")
             self._log_error(exp_id, "Evaluation shape mismatch")
             self.results.append({'experiment': exp_id, 'status': 'eval_failed', 'test_loss': loss, 'test_accuracy': acc, 'error': 'Shape mismatch'})
             return

        labels_for_metrics = list(range(num_classes_eval))

        report = classification_report(
             y_test_class, y_pred_class,
             labels=labels_for_metrics,
             target_names=class_names,
             output_dict=True,
             zero_division=0
         )
        cm = confusion_matrix(y_test_class, y_pred_class, labels=labels_for_metrics)

        report_df = pd.DataFrame(report).transpose()
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        report_df.to_csv(self.config.RESULTS_PATH / f"{exp_id}_report.csv")
        cm_df.to_csv(self.config.RESULTS_PATH / f"{exp_id}_cm.csv")
        print(f"[Evaluation] Saved classification report and confusion matrix to {self.config.RESULTS_PATH}")

        self.results.append({
            'experiment': exp_id,
            'status': 'success',
            'test_loss': loss,
            'test_accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'error': None
        })

    def _save_run_artifacts(self, exp_id, scaler, encoder):
        dump(scaler, self.config.SCALERS_PATH / f"{exp_id}_scaler.joblib")
        if encoder:
            dump(encoder, self.config.ENCODERS_PATH / f"{exp_id}_encoder.joblib")

    def _save_overall_best_model(self):
        if self.best_model_exp_id and self.best_model_path:
            print(f"\n--- Saving Overall Best Model (based on validation loss) ---")
            print(f"Best Experiment ID: {self.best_model_exp_id}")
            print(f"Best Validation Loss: {self.best_val_loss:.4f}")

            overall_best_model_dest = self.config.ARTIFACTS_PATH / "overall_best_transformer_model.h5"
            overall_best_scaler_dest = self.config.ARTIFACTS_PATH / "overall_best_transformer_scaler.joblib"
            overall_best_encoder_dest = self.config.ARTIFACTS_PATH / "overall_best_transformer_encoder.joblib"

            try:
                if self.best_model_path.exists():
                    shutil.copyfile(self.best_model_path, overall_best_model_dest)
                    print(f"Saved best model to: {overall_best_model_dest}")
                else:
                    print(f"Warning: Best model checkpoint not found at {self.best_model_path}")

                if self.best_scaler_path and self.best_scaler_path.exists():
                     shutil.copyfile(self.best_scaler_path, overall_best_scaler_dest)
                     print(f"Saved best scaler to: {overall_best_scaler_dest}")
                else:
                     print(f"Warning: Best scaler not found at {self.best_scaler_path}")

                if self.best_encoder_path and self.best_encoder_path.exists():
                     shutil.copyfile(self.best_encoder_path, overall_best_encoder_dest)
                     print(f"Saved best encoder to: {overall_best_encoder_dest}")
                elif self.best_encoder_path:
                     print(f"Warning: Best encoder not found at {self.best_encoder_path}")

            except Exception as e:
                print(f"Error copying best model artifacts: {e}")
        else:
            print("\n--- No best model identified (perhaps all runs failed or 'val_loss' was missing) ---")

    def _save_plots(self, exp_id, history, model, X_test_seq, y_test_seq, encoder):
        plt.figure(figsize=(12, 5))
        try:
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'{exp_id} Accuracy Curves')
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'{exp_id} Loss Curves')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.config.PLOTS_PATH / f"{exp_id}_learning_curves.png")
            plt.close()
            print(f"[Plotting] Saved learning curves to {self.config.PLOTS_PATH}")
        except KeyError as e:
            print(f"Warning: Could not plot learning curves, missing key: {e}")
        except Exception as e:
            print(f"Warning: Error saving learning curves plot: {e}")
            plt.close()

        try:
            y_pred = model.predict(X_test_seq)
            if model.output_shape[-1] == 1: # Binary
                 y_pred_class = (y_pred > 0.5).astype(int).flatten()
                 y_test_class = y_test_seq.flatten() if y_test_seq.ndim > 1 else y_test_seq
                 class_names = ['Benign', 'Attack']
                 num_classes_plot = 2
            else: # Multi-class
                 y_pred_class = np.argmax(y_pred, axis=1)
                 y_test_class = y_test_seq
                 num_classes_plot = model.output_shape[-1]
                 if encoder is not None and hasattr(encoder, 'classes_'):
                     class_names = encoder.classes_
                 else:
                     class_names = [str(i) for i in range(num_classes_plot)]

            if y_test_class.shape != y_pred_class.shape:
                 warnings.warn(f"Shape mismatch for CM plot: y_test {y_test_class.shape}, y_pred {y_pred_class.shape}. Skipping plot.")
                 return

            labels_for_cm = list(range(num_classes_plot))
            cm = confusion_matrix(y_test_class, y_pred_class, labels=labels_for_cm)
            plt.figure(figsize=(max(8, num_classes_plot * 0.8), max(6, num_classes_plot * 0.6)))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'{exp_id} Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(self.config.PLOTS_PATH / f"{exp_id}_confusion_matrix.png")
            plt.close()
            print(f"[Plotting] Saved confusion matrix plot to {self.config.PLOTS_PATH}")
        except Exception as e:
            print(f"Warning: Error saving confusion matrix plot: {e}")
            import traceback
            traceback.print_exc()
            plt.close()

    def _log_error(self, exp_id, error_message):
        self.results.append({
            'experiment': exp_id,
            'status': 'failed',
            'test_loss': None,
            'test_accuracy': None,
            'macro_f1': None,
            'weighted_f1': None,
            'error': error_message
        })

    def _save_results(self):
        results_df = pd.DataFrame(self.results)
        output_file = self.config.RESULTS_PATH / 'transformer_experiment_summary.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n=== All Transformer experiments finished. Results saved to: {output_file} ===")


if __name__ == "__main__":
    print("Starting Transformer Experiment Runner...")
    if 'train_df' in locals() and 'test_df' in locals():
         runner = TransformerExperimentRunner(train_df, test_df)
         runner.run_experiments()
    else:
         print("Failed to load data. Exiting.")
