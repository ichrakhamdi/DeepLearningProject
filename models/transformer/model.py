import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding, GlobalAveragePooling1D
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class TransformerModel:
    def __init__(self, input_shape, num_classes,
                 num_heads=8, key_dim=64, ff_dim=128,
                 num_transformer_blocks=2, dropout_rate=0.1):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate

        self.d_model = input_shape[1]
        self.window_size = input_shape[0]

        self.model = self._build_model()

    def _transformer_encoder(self, inputs):
        x = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)(inputs, inputs)
        x = Dropout(self.dropout_rate)(x)
        res = x + inputs
        x = LayerNormalization(epsilon=1e-6)(res)

        # Feed Forward Part
        ff_out = Dense(self.ff_dim, activation="relu")(x)
        ff_out = Dropout(self.dropout_rate)(ff_out)
        ff_out = Dense(self.d_model)(ff_out) # Project back to input feature dimension
        ff_out = Dropout(self.dropout_rate)(ff_out)

        # Add and Norm
        x = LayerNormalization(epsilon=1e-6)(x + ff_out)
        return x

    def _build_model(self):
        inputs = Input(shape=self.input_shape)

        x = inputs

        pos_encoding_layer = PositionalEncoding(self.window_size, self.d_model)
        x = pos_encoding_layer(x)
        x = Dropout(self.dropout_rate)(x)

        for _ in range(self.num_transformer_blocks):
            x = self._transformer_encoder(x)

        x = GlobalAveragePooling1D(data_format="channels_last")(x)
        x = Dropout(self.dropout_rate)(x)

        activation = "sigmoid" if self.num_classes == 1 else "softmax"
        outputs = Dense(self.num_classes, activation=activation)(x)

        return Model(inputs=inputs, outputs=outputs)

