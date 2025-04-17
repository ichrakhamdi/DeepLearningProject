import torch
import torch.nn as nn
import math
from config.settings import TransformerSettings

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.config = TransformerSettings()
        
        # Input shape is (batch_size, seq_len, features)
        self.seq_len = input_shape[0]
        self.n_features = input_shape[1]
        
        # Feature projection
        self.feature_projection = nn.Linear(self.n_features, self.config.D_MODEL)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.config.D_MODEL)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.D_MODEL,
            nhead=self.config.NUM_HEADS,
            dim_feedforward=self.config.DFF,
            dropout=self.config.DROPOUT_RATE,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.NUM_LAYERS
        )
        
        # Final layers
        self.dropout = nn.Dropout(self.config.DROPOUT_RATE)
        self.fc1 = nn.Linear(self.config.D_MODEL, self.config.DENSE_UNITS)
        self.fc2 = nn.Linear(self.config.DENSE_UNITS, num_classes if num_classes > 1 else 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() if num_classes == 1 else None
        
    def forward(self, x):
        # Project features to transformer dimension
        x = self.feature_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Final dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Apply sigmoid for binary classification
        if self.sigmoid is not None:
            x = self.sigmoid(x)
            
        return x 