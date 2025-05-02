import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Conv1D, MaxPooling1D, BatchNormalization

class CNNTransformerModel:
    def __init__(self, input_shape, num_classes,
                 # CNN params
                 filters=64, kernel_size=3, pool_size=2,
                 # Transformer params
                 num_heads=8, key_dim=64, ff_dim=128, num_transformer_blocks=2,
                 dropout_rate=0.1):

        self.input_shape = input_shape
        self.num_classes = num_classes
        # CNN
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        # Transformer
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate

        self.model = self._build_model()

    def _transformer_encoder(self, inputs):
        # Attention and Normalization
        x = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)(inputs, inputs)
        x = Dropout(self.dropout_rate)(x)
        res = x + inputs
        x = LayerNormalization(epsilon=1e-6)(res)

        # Feed Forward Part
        ff_out = Dense(self.ff_dim, activation="relu")(x)
        ff_out = Dropout(self.dropout_rate)(ff_out)
        ff_out = Dense(inputs.shape[-1])(ff_out) # Project back to input dimension
        ff_out = Dropout(self.dropout_rate)(ff_out)

        # Add and Norm
        x = LayerNormalization(epsilon=1e-6)(x + ff_out)
        return x

    def _build_model(self):
        inputs = Input(shape=self.input_shape)

        # CNN Feature Extractor Block
        # You can add more Conv1D/MaxPooling1D layers
        x = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', padding='same')(inputs)
        # Optional: Batch Normalization
        # x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=self.pool_size, padding='same')(x)
        x = Dropout(self.dropout_rate)(x)

        # Optional: Add another CNN block
        # x = Conv1D(filters=self.filters*2, kernel_size=self.kernel_size, activation='relu', padding='same')(x)
        # x = MaxPooling1D(pool_size=self.pool_size, padding='same')(x)
        # x = Dropout(self.dropout_rate)(x)

        # Transformer Blocks
        # Note: The output shape of CNN might change sequence length/feature dim.
        # Ensure the Transformer part handles this shape.
        # Positional encoding might be added here if sequence order is still important after CNN.

        for _ in range(self.num_transformer_blocks):
            x = self._transformer_encoder(x)

        # Pooling and Classification Head
        x = GlobalAveragePooling1D(data_format="channels_last")(x)
        x = Dropout(self.dropout_rate)(x)

        activation = "sigmoid" if self.num_classes == 1 else "softmax"
        outputs = Dense(self.num_classes, activation=activation)(x)

        return Model(inputs=inputs, outputs=outputs)

# Example Usage (requires data preprocessing steps):
# input_shape = (window_size, num_features) # e.g., (10, 78)
# num_classes = 1 # For binary classification
# cnn_transformer = CNNTransformerModel(input_shape, num_classes)
# model = cnn_transformer.model
# model.summary()
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
