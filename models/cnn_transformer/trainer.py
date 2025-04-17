import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy, SparseCategoricalAccuracy
from models.TFTransformer.cnn_model import CNNTFTransformerModel
from config.CNNTFTransformer_settings import CNNTFTransformerSettings
import tensorflow_addons as tfa

class CNNTFTransformerTrainer:
    def __init__(self, experiment_id):
        self.config = CNNTFTransformerSettings()
        self.experiment_id = experiment_id

    def train(self, X_train, y_train, X_val, y_val, num_classes, class_type, class_weights=None):
        # Build model and tf.data.Datasets
        cnn_tf = CNNTFTransformerModel(X_train, y_train, X_val, y_val, num_classes, class_type)
        model = cnn_tf.model
        train_dataset = cnn_tf.train_dataset
        val_dataset = cnn_tf.val_dataset

        # Optimizer with optional weight decay
        weight_decay = getattr(self.config, 'WEIGHT_DECAY', None)
        optimizer = (tfa.optimizers.AdamW(learning_rate=self.config.LEARNING_RATE,
                                          weight_decay=weight_decay)
                     if weight_decay is not None else tf.keras.optimizers.Adam(self.config.LEARNING_RATE))

        # Loss and metrics
        loss = (tf.keras.losses.BinaryCrossentropy() if num_classes == 1
                else tf.keras.losses.SparseCategoricalCrossentropy())
        metrics = ([BinaryAccuracy(name='binary_accuracy')] if num_classes == 1
                   else [SparseCategoricalAccuracy(name='accuracy')])

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Callbacks
        ckpt_dir = os.path.join(self.config.MODELS_PATH, f'ckpt_{self.experiment_id}')
        os.makedirs(ckpt_dir, exist_ok=True)
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.PATIENCE,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=self.config.PATIENCE // 2
            ),
            ModelCheckpoint(
                filepath=os.path.join(ckpt_dir, f'best_weights_{self.experiment_id}'),
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                save_weights_only=True,
                verbose=1
            )
        ]

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        return model, history 