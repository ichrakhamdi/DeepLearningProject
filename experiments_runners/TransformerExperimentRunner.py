import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from joblib import dump
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
import shutil
from experiments_runners.BaseExperimentRunner import BaseExperimentRunner
from config.Trasformer_settings import TransformerSettings
from data.code.preprocessor import DataPreprocessor
from models.transformer.trainer import TransformerTrainer

class TransformerExperimentRunner(BaseExperimentRunner):
    def __init__(self, train_data, test_data):
        super().__init__(train_data, test_data)
        self.config = TransformerSettings()
        self.best_val_loss = float('inf')
        self.best_model_exp_id = None
        self.best_model_path = None
        self.best_scaler_path = None
        self.best_encoder_path = None
        self.current_experiment_num_classes = None

    def _run_single_experiment(self, class_type, aug, cw):
        exp_id = f"transformer_{class_type}_aug{aug}_cw{cw}_win{self.config.WINDOW_SIZE}"
        print(f"\n=== Running Transformer experiment: {exp_id} ===")

        try:
            # Split train data into train and validation
            train_df_split, val_df_split = train_test_split(
                self.train_df,
                test_size=0.2,
                random_state=42,
                stratify=self.train_df[self.config.CLASS_MAP[class_type]] if class_type in self.config.CLASS_MAP else None
            )
            print(f"[Data Split] Train: {len(train_df_split)}, Validation: {len(val_df_split)}, Test: {len(self.test_df)}")

            # Preprocess train data (with augmentation if requested)
            X_train, y_train, encoder = self.preprocessor.preprocess_train(
                train_df_split, class_type, augmentation=aug
            )
            print(f"[Preprocessing Train] Shape after scaling/encoding (and augmentation={aug}): X={X_train.shape}, y={y_train.shape}")

            # Preprocess validation data (no augmentation)
            X_val, y_val, _ = self.preprocessor.preprocess_validation_test(
                val_df_split, class_type
            )
            print(f"[Preprocessing Val] Shape after scaling/encoding: X={X_val.shape}, y={y_val.shape}")

            # Preprocess test data (no augmentation)
            X_test, y_test, _ = self.preprocessor.preprocess_validation_test(
                self.test_df, class_type
            )
            print(f"[Preprocessing Test] Shape after scaling/encoding: X={X_test.shape}, y={y_test.shape}")

            # Create sequences for Transformer
            window_size = self.config.WINDOW_SIZE
            print(f"[Sequence Creation] Using window size: {window_size}")

            if len(X_train) < window_size or len(X_val) < window_size or len(X_test) < window_size:
                warnings.warn(f"Skipping experiment {exp_id}: Not enough data points for window size {window_size}.")
                self._log_error(exp_id, f"Insufficient data for window size {window_size}")
                return

            X_train_seq, y_train_seq = self.preprocessor.create_sequences(X_train, y_train, window_size)
            X_val_seq, y_val_seq = self.preprocessor.create_sequences(X_val, y_val, window_size)
            X_test_seq, y_test_seq = self.preprocessor.create_sequences(X_test, y_test, window_size)
            print(f"[Sequence Shapes] Train: X={X_train_seq.shape}, y={y_train_seq.shape} | Val: X={X_val_seq.shape}, y={y_val_seq.shape} | Test: X={X_test_seq.shape}, y={y_test_seq.shape}")

            if X_train_seq.shape[0] == 0 or X_val_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
                warnings.warn(f"Skipping experiment {exp_id}: Empty sequences after applying window size {window_size}.")
                self._log_error(exp_id, f"Empty sequences for window size {window_size}")
                return

            # Determine number of classes and class weights
            num_classes = 1 if class_type == 'binary' else len(np.unique(y_train_seq))
            self.current_experiment_num_classes = num_classes
            print(f"[Model Params] num_classes: {num_classes}")
            class_weights = self.preprocessor.get_class_weights(y_train_seq) if cw else None
            if class_weights:
                print(f"[Model Params] Applying class weights: {class_weights}")

            # Train model
            trainer = TransformerTrainer(experiment_id=exp_id)
            print(f"[Training] Starting training for {self.config.EPOCHS} epochs (Batch size: {self.config.BATCH_SIZE})...")
            model, history = trainer.train(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                num_classes,
                class_weights
            )
            print("[Training] Training finished.")

            # Track best model based on validation loss
            try:
                current_min_val_loss = min(history.history['val_loss'])
                print(f"[Tracking] Min validation loss for this run: {current_min_val_loss:.4f}")
                if current_min_val_loss < self.best_val_loss:
                    self.best_val_loss = current_min_val_loss
                    self.best_model_exp_id = exp_id
                    self.best_model_path = self.config.MODELS_PATH / f"{exp_id}_model.h5"
                    self.best_scaler_path = self.config.SCALERS_PATH / f"{exp_id}_scaler.joblib"
                    self.best_encoder_path = self.config.ENCODERS_PATH / f"{exp_id}_encoder.joblib" if encoder else None
                    print(f"*** New best model found! exp_id: {exp_id}, val_loss: {self.best_val_loss:.4f} ***")
            except (KeyError, ValueError) as e:
                warnings.warn(f"Could not track best model for {exp_id}: {str(e)}")

            # Evaluate and save results
            print("[Evaluation] Evaluating model on test set...")
            self._evaluate_and_save(exp_id, model, X_test_seq, y_test_seq, encoder)

            # Save artifacts and plots
            print("[Saving] Saving artifacts and plots for this run...")
            self._save_artifacts(exp_id, model, encoder)
            self._save_plots(exp_id, history, model, X_test_seq, y_test_seq, encoder)
            print(f"=== Experiment {exp_id} completed successfully ===")

        except Exception as e:
            print(f"!!! Experiment {exp_id} failed: {str(e)} !!!")
            import traceback
            traceback.print_exc()
            self._log_error(exp_id, str(e))

    def _evaluate_and_save(self, exp_id, model, X_test_seq, y_test_seq, encoder):
        # Evaluate model
        eval_results = model.evaluate(X_test_seq, y_test_seq, verbose=0)
        loss = eval_results[0]
        acc = eval_results[1]
        print(f"[Evaluation] Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

        # Predict on test set
        y_pred = model.predict(X_test_seq)

        # Handle binary vs. multi-class classification
        if model.output_shape[-1] == 1:  # Binary classification
            y_pred_class = (y_pred > 0.5).astype(int).flatten()
            y_test_class = y_test_seq.flatten() if y_test_seq.ndim > 1 else y_test_seq
            class_names = ['Benign', 'Attack']
            num_classes = 2
        else:  # Multi-class classification
            y_pred_class = np.argmax(y_pred, axis=1)
            y_test_class = y_test_seq
            num_classes = model.output_shape[-1]
            class_names = encoder.classes_ if encoder and hasattr(encoder, 'classes_') else [str(i) for i in range(num_classes)]
            print(f"[Evaluation] Multi-class detected. Target classes: {class_names}")

        # Check for shape mismatch
        if y_test_class.shape != y_pred_class.shape:
            warnings.warn(f"Shape mismatch for evaluation: y_test {y_test_class.shape}, y_pred {y_pred_class.shape}")
            self._log_error(exp_id, "Evaluation shape mismatch")
            self.results.append({
                'experiment': exp_id,
                'status': 'eval_failed',
                'test_loss': loss,
                'test_accuracy': acc,
                'error': 'Shape mismatch'
            })
            return

        # Generate classification report and confusion matrix
        labels = list(range(num_classes))
        report = classification_report(
            y_test_class, y_pred_class,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        cm = confusion_matrix(y_test_class, y_pred_class, labels=labels)

        # Save report and confusion matrix
        pd.DataFrame(report).transpose().to_csv(self.config.RESULTS_PATH / f"{exp_id}_report.csv")
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(self.config.RESULTS_PATH / f"{exp_id}_cm.csv")
        print(f"[Evaluation] Saved classification report and confusion matrix to {self.config.RESULTS_PATH}")

        # Append results
        self.results.append({
            'experiment': exp_id,
            'status': 'success',
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'test_loss': loss
        })

    def _save_plots(self, exp_id, history, model, X_test_seq, y_test_seq, encoder):
        # Plot accuracy and loss curves
        plt.figure(figsize=(12, 5))
        try:
            plt.subplot(1, 2, 1)
            metric_key = 'accuracy' if 'accuracy' in history.history else 'output_accuracy'
            val_metric_key = f'val_{metric_key}'
            plt.plot(history.history[metric_key], label='Train Accuracy')
            plt.plot(history.history[val_metric_key], label='Validation Accuracy')
            plt.title(f'{exp_id} Accuracy Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'{exp_id} Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.config.PLOTS_PATH / f"{exp_id}_curves.png")
            plt.close()
            print(f"[Plotting] Saved learning curves to {self.config.PLOTS_PATH}")
        except KeyError as e:
            print(f"Warning: Could not plot learning curves, missing key: {e}")
            plt.close()

        # Plot confusion matrix
        try:
            y_pred = model.predict(X_test_seq)
            if model.output_shape[-1] == 1:  # Binary classification
                y_pred_class = (y_pred > 0.5).astype(int).flatten()
                y_test_class = y_test_seq.flatten() if y_test_seq.ndim > 1 else y_test_seq
                class_names = ['Benign', 'Attack']
                num_classes = 2
            else:  # Multi-class classification
                y_pred_class = np.argmax(y_pred, axis=1)
                y_test_class = y_test_seq
                num_classes = model.output_shape[-1]
                class_names = encoder.classes_ if encoder and hasattr(encoder, 'classes_') else [str(i) for i in range(num_classes)]

            if y_test_class.shape != y_pred_class.shape:
                warnings.warn(f"Shape mismatch for CM plot: y_test {y_test_class.shape}, y_pred {y_pred_class.shape}")
                return

            labels = list(range(num_classes))
            cm = confusion_matrix(y_test_class, y_pred_class, labels=labels)
            plt.figure(figsize=(max(8, num_classes * 0.8), max(6, num_classes * 0.6)))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'{exp_id} Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(self.config.PLOTS_PATH / f"{exp_id}_cm.png")
            plt.close()
            print(f"[Plotting] Saved confusion matrix plot to {self.config.PLOTS_PATH}")
        except Exception as e:
            print(f"Warning: Error saving confusion matrix plot: {e}")
            plt.close()

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