from pathlib import Path
import warnings
import pandas as pd

class LSTMSettings:
    # Data configuration
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
    
    # Experiment parameters
    CLASS_TYPES = ['binary', '6class', '19class']
    AUGMENTATIONS = [True, False]
    CLASS_WEIGHTS = [True, False]
    WINDOW_SIZE = 4
    
    # Model configuration
    LSTM_UNITS = 128
    DROPOUT_RATE = 0.4
    LEARNING_RATE = 0.001
    BATCH_SIZE = 1024
    EPOCHS = 50
    PATIENCE = 5
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    ARTIFACTS_PATH = BASE_DIR / 'artifacts' / 'lstm'
    MODELS_PATH = ARTIFACTS_PATH / 'models'
    SCALERS_PATH = ARTIFACTS_PATH / 'scalers'
    ENCODERS_PATH = ARTIFACTS_PATH / 'encoders'
    PLOTS_PATH = ARTIFACTS_PATH / 'plots'
    RESULTS_PATH = ARTIFACTS_PATH / 'results'
    
    def __init__(self):
        for path in [self.MODELS_PATH, self.SCALERS_PATH, 
                     self.ENCODERS_PATH, self.PLOTS_PATH, self.RESULTS_PATH]:
            path.mkdir(parents=True, exist_ok=True)
        
        train_file = Path("data/concatenated/train.csv")
        if train_file.exists():
            train_df = pd.read_csv(train_file)
            unique_labels = train_df[self.CLASS_MAP['binary']].unique()
            if 'Benign' not in unique_labels and len(unique_labels) == 2:
                warnings.warn("Binary labels do not contain 'Benign'. Will assume the first sorted label as benign for binary classification.")
            elif len(unique_labels) != 2:
                raise ValueError("Binary classification requires exactly two classes in the label column.")
