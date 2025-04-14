import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Accuracy
from models.TFTransformer.model import TFTransformerModel
from config.TFTrasformer_settings import TFTransformerSettings
import tensorflow_addons as tfa

import os


class TFTransformerTrainer:
    def __init__(self, experiment_id):
        self.tf_transformer_config = TFTransformerSettings()
        self.experiment_id = experiment_id

    def train(self, X_train, y_train, X_val, y_val, num_classes, class_type, class_weights):

        tf_transform = TFTransformerModel(X_train, y_train, X_val, y_val, num_classes, class_type)
        model = tf_transform.model
        train_dataset = tf_transform.train_dataset
        val_dataset = tf_transform.val_dataset

        model.compile(

            optimizer=tfa.optimizers.AdamW(learning_rate=self.tf_transformer_config.LEARNING_RATE,
                                           weight_decay=self.tf_transformer_config.WEIGHT_DECAY),
            loss=self._get_loss(num_classes),
            metrics=self._get_metrics(num_classes),
        )

        callbacks = self._get_callbacks()

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.tf_transformer_config.EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        return model, history

    def _get_loss(self, num_classes):
        if num_classes == 1:
            # Binary classification: sigmoid output
            return {"output": tf.keras.losses.BinaryCrossentropy(from_logits=False), "importances": None}
        else:
            # Multi-class classification: softmax output
            return {"output": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), "importances": None}

    def _get_metrics(self, num_classes):
        base_metrics = [Accuracy(name="accuracy")]
        if num_classes == 1:
            output_metrics = {"output": [tf.keras.metrics.Accuracy(name="accuracy")], "importances": None}
        else:
            output_metrics = {"output": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
                              "importances": None}
        return output_metrics

    def _get_callbacks(self):
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=self.tf_transformer_config.PATIENCE,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=self.tf_transformer_config.PATIENCE // 2
            ),

            ModelCheckpoint(
                filepath=os.path.join(self.tf_transformer_config.MODELS_PATH, f'ckpt_{self.experiment_id}',
                                      f'best_weights_{self.experiment_id}'),
                save_best_only=True,
                mode='min',
                monitor='loss',
                save_weights_only=True,
                verbose=1
            )

        ]
