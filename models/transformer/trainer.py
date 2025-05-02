# Ensure necessary imports
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, AUC

# Adjust settings import to use the new Transformer settings
from config.transformer_settings import TransformerSettings
from models.transformer.model import TransformerModel

class TransformerTrainer:
    def __init__(self, experiment_id):
        # Use TransformerSettings
        self.config = TransformerSettings()
        self.experiment_id = experiment_id

    def train(self, X_train, y_train, X_val, y_val, num_classes, class_weights):
        if len(X_train.shape) != 3:
            raise ValueError(f"Input shape must be 3D (samples, timesteps, features). Got {X_train.shape}")

        # Instantiate the TransformerModel using parameters from TransformerSettings
        model = TransformerModel(
            input_shape=X_train.shape[1:],
            num_classes=num_classes,
            num_heads=self.config.NUM_HEADS,
            key_dim=self.config.KEY_DIM,
            ff_dim=self.config.FF_DIM,
            num_transformer_blocks=self.config.NUM_TRANSFORMER_BLOCKS,
            dropout_rate=self.config.DROPOUT_RATE
        ).model

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss=self._get_loss(num_classes),
            metrics=self._get_metrics(num_classes)
        )

        callbacks = self._get_callbacks()

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        return model, history

    def _get_loss(self, num_classes):
        # Loss function depends on the output layer activation and label format
        if num_classes == 1: # Assuming sigmoid output for binary
            return tf.keras.losses.BinaryCrossentropy()
        else: # Assuming softmax output for multi-class
            # Use SparseCategoricalCrossentropy if labels are integers
            return tf.keras.losses.SparseCategoricalCrossentropy()
            # Use CategoricalCrossentropy if labels are one-hot encoded
            # return tf.keras.losses.CategoricalCrossentropy()

    def _get_metrics(self, num_classes):
        base_metrics = ['accuracy']
        if num_classes == 1:
            return base_metrics + [
                AUC(name='auc'),
                Precision(name='precision'),
                Recall(name='recall')
            ]
        # For sparse multi-class labels
        return base_metrics + [
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='top1_acc')
            # Add other relevant metrics if needed
        ]
        # If using one-hot encoded labels for multi-class:
        # return base_metrics + [tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top1_acc')]


    def _get_callbacks(self):
        # Use MODELS_PATH from TransformerSettings
        checkpoint_path = self.config.MODELS_PATH / f'best_transformer_model_{self.experiment_id}.h5'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        return [
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
                filepath=str(checkpoint_path),
                save_best_only=True,
                monitor='val_loss'
            )
        ]
