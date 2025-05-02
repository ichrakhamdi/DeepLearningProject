from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from config.LSTM_settings import LSTMSettings

class LSTMModel:
    def __init__(self, input_shape, num_classes):
        self.lstm_config = LSTMSettings()
        self.model = self._build_model(input_shape, num_classes)
        
    def _build_model(self, input_shape, num_classes):
        model = Sequential([
            LSTM(self.lstm_config.LSTM_UNITS, return_sequences=True, input_shape=input_shape),
            Dropout(self.lstm_config.DROPOUT_RATE),
            LSTM(self.lstm_config.LSTM_UNITS * 2),
            Dropout(self.lstm_config.DROPOUT_RATE),
            Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
        ])
        return model
