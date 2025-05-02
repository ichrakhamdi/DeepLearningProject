from pathlib import Path
import warnings
import pandas as pd
from config.general_settings import GeneralSettings

class TransformerSettings(GeneralSettings):
    # Model configuration
    NUM_HEADS = 8       # Number of attention heads
    KEY_DIM = 64        # Dimension of key/query/value per head
    FF_DIM = 128        # Hidden layer size in feed forward network inside transformer
    NUM_TRANSFORMER_BLOCKS = 2 # Number of transformer encoder blocks
    DROPOUT_RATE = 0.1
    LEARNING_RATE = 0.001
    BATCH_SIZE = 1024
    EPOCHS = 20
    PATIENCE = 5

    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    ARTIFACTS_PATH = BASE_DIR / 'artifacts' / 'transformer'
    MODELS_PATH = ARTIFACTS_PATH / 'models'
    SCALERS_PATH = ARTIFACTS_PATH / 'scalers'
    ENCODERS_PATH = ARTIFACTS_PATH / 'encoders'
    PLOTS_PATH = ARTIFACTS_PATH / 'plots'
    RESULTS_PATH = ARTIFACTS_PATH / 'results'

    def __init__(self):
        # Create necessary directories
        for path in [self.MODELS_PATH, self.SCALERS_PATH,
                     self.ENCODERS_PATH, self.PLOTS_PATH, self.RESULTS_PATH]:
            path.mkdir(parents=True, exist_ok=True)

        # Validate binary classification labels
        train_file = Path("data/concatenated/train.csv")
        if train_file.exists():
            try:
                train_df = pd.read_csv(train_file, usecols=[self.CLASS_MAP['binary']])
                unique_labels = train_df[self.CLASS_MAP['binary']].unique()
                if 'Benign' not in unique_labels and len(unique_labels) == 2:
                    warnings.warn(
                        "Binary labels do not contain 'Benign'. Will assume the first sorted label as benign for binary classification.")
                elif len(unique_labels) != 2:
                    raise ValueError("Binary classification requires exactly two classes in the label column.")
            except Exception as e:
                warnings.warn(f"Could not perform initial data validation check: {e}")
        else:
            warnings.warn(f"Train file not found at {train_file}, skipping initial validation.")