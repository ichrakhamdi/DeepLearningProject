import pandas as pd
import numpy as np
import tensorflow as tf
from config.CNNTFTransformer_settings import CNNTFTransformerSettings
from .utils import df_to_dataset
from tabtransformertf.models.fttransformer import FTTransformerEncoder, FTTransformer
from tensorflow.keras.layers import Input, Conv1D, ReLU, GlobalAveragePooling1D
from tensorflow.keras.models import Model

class CNNTFTransformerModel:
    def __init__(self, X_train, y_train, X_val, y_val, num_classes, class_type):
        self.config = CNNTFTransformerSettings()
        # prepare datasets same as TFTransformerModel
        train_df = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis=1),
                                columns=self.config.FEATURE_COLS + [class_type])
        val_df = pd.DataFrame(np.concatenate((X_val, y_val.reshape(-1,1)), axis=1),
                              columns=self.config.FEATURE_COLS + [class_type])
        # tf.data.Datasets
        self.train_dataset = df_to_dataset(train_df, class_type,
                                           shuffle=True, batch_size=self.config.PATIENCE)
        self.val_dataset = df_to_dataset(val_df, class_type,
                                         shuffle=True, batch_size=self.config.PATIENCE)
        # build model
        self.model = self._build_model(len(self.config.FEATURE_COLS), num_classes)

    def _build_model(self, num_features, num_classes):
        # CNN front-end
        inputs = Input(shape=(num_features, 1), name='features')
        x = Conv1D(filters=self.config.embedding_dim,
                   kernel_size=self.config.CNN_KERNEL_SIZE,
                   padding='same')(inputs)
        x = ReLU()(x)
        x = GlobalAveragePooling1D()(x)  # (batch, embedding_dim)

        # Pass through FTTransformerEncoder on original features
        ft_encoder = FTTransformerEncoder(
            numerical_features=self.config.FEATURE_COLS,
            categorical_features=[],
            numerical_data=None,
            categorical_data=None,
            y=None,
            numerical_embedding_type='linear',
            embedding_dim=self.config.embedding_dim,
            depth=self.config.depth,
            heads=self.config.heads,
            attn_dropout=self.config.attn_dropout,
            ff_dropout=self.config.ff_dropout,
            explainable=self.config.explainable,
        )
        ft_model = FTTransformer(
            encoder=ft_encoder,
            out_dim=num_classes,
            out_activation='sigmoid' if num_classes == 1 else 'softmax'
        )

        # Combine: treat x as new single feature and feed into FTTransformer
        # NOTE: FTTransformer expects dict-input, so we wrap ft_model here
        outputs = ft_model(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model 