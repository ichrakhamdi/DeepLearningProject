from config.TFTrasformer_settings import TFTransformerSettings
import pandas as pd
import numpy as np

from .utils import df_to_dataset
from tabtransformertf.models.fttransformer import FTTransformerEncoder, FTTransformer


class TFTransformerModel:
    def __init__(self, X_train, y_train, X_val, y_val, num_classes, class_type):
        self.tf_config = TFTransformerSettings()
        self.model = self._build_model(X_train, y_train, X_val, y_val, num_classes, class_type)

    def _build_model(self, X_train, y_train, X_val, y_val, num_classes, class_type):
        train_df_after_scaling = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1),
                                              columns=self.tf_config.FEATURE_COLS + [class_type])

        val_df_after_scaling = pd.DataFrame(np.concatenate((X_val, y_val.reshape(-1, 1)), axis=1),
                                            columns=self.tf_config.FEATURE_COLS + [class_type])

        train_df_after_scaling[class_type] = train_df_after_scaling[class_type].to_numpy()

        self.train_dataset = df_to_dataset(train_df_after_scaling, class_type,
                                           shuffle=True, batch_size=1024)
        self.val_dataset = df_to_dataset(val_df_after_scaling, class_type,
                                         shuffle=True, batch_size=1024)

        ft_linear_encoder = FTTransformerEncoder(
            numerical_features=self.tf_config.FEATURE_COLS,  # list of numeric features
            categorical_features=[],  # list of numeric features
            numerical_data=train_df_after_scaling[self.tf_config.FEATURE_COLS].values,
            categorical_data=None,
            y=None,
            numerical_embedding_type='linear',
            embedding_dim=self.tf_config.embedding_dim,
            depth=self.tf_config.depth,
            heads=self.tf_config.heads,
            attn_dropout=self.tf_config.attn_dropout,
            ff_dropout=self.tf_config.ff_dropout,
            explainable=self.tf_config.explainable,
        )
        # Pass the encoder to the model
        ft_model = FTTransformer(
            encoder=ft_linear_encoder,
            out_dim=num_classes,  # Number of outputs in final layer
            out_activation='sigmoid' if num_classes == 1 else 'softmax',  # Activation function for final layer
        )
        return ft_model
