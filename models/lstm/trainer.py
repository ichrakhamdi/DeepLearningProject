import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, AUC
from config.settings import LSTMSettings
from models.lstm.model import LSTMModel

class LSTMTrainer:
    def __init__(self, experiment_id):
        self.config = LSTMSettings()
        self.experiment_id = experiment_id
        
    def train(self, X_train, y_train, X_val, y_val, num_classes, class_weights):

        checkpoint_path = self.config.MODELS_PATH / f'best_model_{self.experiment_id}.h5'

        if checkpoint_path.exists():
            print(f"Loading existing checkpoint from {checkpoint_path} ...")
            model = tf.keras.models.load_model(str(checkpoint_path))
            additional_epochs = 5
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=additional_epochs,
                batch_size=self.config.BATCH_SIZE,
                class_weight=class_weights,
                verbose=1
            )
            return model, history

        # If no checkpoint exists, build a new model
        model = LSTMModel(X_train.shape[1:], num_classes).model

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
        if num_classes == 1:
            return tf.keras.losses.BinaryCrossentropy()
        elif num_classes == 2:
            return tf.keras.losses.SparseCategoricalCrossentropy()
        return tf.keras.losses.SparseCategoricalCrossentropy()
    
    def _get_metrics(self, num_classes):
        base_metrics = ['accuracy']
        if num_classes == 1:
            return base_metrics + [
                AUC(name='auc'),
                Precision(name='precision'),
                Recall(name='recall')
            ]
        return base_metrics + [
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='top1_acc')
        ]
    
    def _get_callbacks(self):
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
                filepath=str(self.config.MODELS_PATH / f'best_model_{self.experiment_id}.h5'),
                save_best_only=True,
                monitor='val_loss'
            )
        ]
