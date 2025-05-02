from pathlib import Path
import warnings
import pandas as pd

class CNNTransformerSettings:
    # Data configuration (shared for now)
    FEATURE_COLS = [
        'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
        'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
        'Std', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance',
        'Variance', 'Weight'
    ]

    CLASS_MAP = {
        'binary': 'label_2',
        '6class': 'label_6',
        '19class': 'label_19'
    }

    # Experiment parameters (shared for now)
    CLASS_TYPES = ['binary', '6class', '19class']
    AUGMENTATIONS = [True, False]
    CLASS_WEIGHTS = [True, False]
    WINDOW_SIZE = 4 # Sequence length input

    # Model configuration
    # CNN part
    CNN_FILTERS = 64
    KERNEL_SIZE = 3
    POOL_SIZE = 2
    # Transformer part
    NUM_HEADS = 4 # Adjusted example
    KEY_DIM = 32  # Adjusted example (ensure consistency with output dim of CNN or add projection)
    FF_DIM = 64   # Adjusted example
    NUM_TRANSFORMER_BLOCKS = 1 # Adjusted example
    # General
    DROPOUT_RATE = 0.1
    LEARNING_RATE = 0.001
    BATCH_SIZE = 1024
    EPOCHS = 20
    PATIENCE = 5

    # Paths - adjusted for CNN+Transformer
    BASE_DIR = Path(__file__).resolve().parent.parent
    ARTIFACTS_PATH = BASE_DIR / 'artifacts' / 'cnn_transformer' # Changed subfolder
    MODELS_PATH = ARTIFACTS_PATH / 'models'
    SCALERS_PATH = ARTIFACTS_PATH / 'scalers'
    ENCODERS_PATH = ARTIFACTS_PATH / 'encoders'
    PLOTS_PATH = ARTIFACTS_PATH / 'plots'
    RESULTS_PATH = ARTIFACTS_PATH / 'results'

    def __init__(self):
        for path in [self.MODELS_PATH, self.SCALERS_PATH,
                     self.ENCODERS_PATH, self.PLOTS_PATH, self.RESULTS_PATH]:
            path.mkdir(parents=True, exist_ok=True)

        # Validation check (optional)
        train_file = Path("data/concatenated/train.csv")
        if train_file.exists():
             try:
                train_df = pd.read_csv(train_file, usecols=[self.CLASS_MAP['binary']])
                unique_labels = train_df[self.CLASS_MAP['binary']].unique()
                if 'Benign' not in unique_labels and len(unique_labels) == 2:
                     warnings.warn("Binary labels do not contain 'Benign'. Will assume the first sorted label as benign for binary classification in preprocessor.")
                elif len(unique_labels) != 2:
                     warnings.warn(f"Expected 2 unique labels for binary classification, but found {len(unique_labels)} in {self.CLASS_MAP['binary']} column.")
             except Exception as e:
                 warnings.warn(f"Could not perform initial data validation check: {e}")
        else:
            warnings.warn(f"Train file not found at {train_file}, skipping initial validation.")
