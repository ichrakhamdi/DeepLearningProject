from pathlib import Path
import warnings
import pandas as pd

class BaseSettings:
    # Common paths
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_PATH = PROJECT_ROOT / 'artifacts' / 'models'
    SCALERS_PATH = PROJECT_ROOT / 'artifacts' / 'scalers'
    ENCODERS_PATH = PROJECT_ROOT / 'artifacts' / 'encoders'
    RESULTS_PATH = PROJECT_ROOT / 'artifacts' / 'results'
    PLOTS_PATH = PROJECT_ROOT / 'artifacts' / 'plots'

    # Create directories if they don't exist
    def __init__(self):
        for path in [self.MODELS_PATH, self.SCALERS_PATH, 
                    self.ENCODERS_PATH, self.PLOTS_PATH, self.RESULTS_PATH]:
            path.mkdir(parents=True, exist_ok=True)

    # Common settings - updated to match your actual data columns
    FEATURE_COLS = [
        'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
        'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
        'ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'ack_count',
        'syn_count', 'fin_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet',
        'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv',
        'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
        'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
    ]

    CLASS_MAP = {
        'binary': 'label_2',
        'multi': 'label_6',
        'full': 'label_19'
    }

    CLASS_TYPES = ['binary', 'multi', 'full']
    AUGMENTATIONS = [True, False]
    CLASS_WEIGHTS = [True, False]

class LSTMSettings(BaseSettings):
    # Model parameters
    HIDDEN_UNITS = 64
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.001
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    PATIENCE = 10

class TransformerSettings(BaseSettings):
    # Model architecture parameters
    D_MODEL = 128  # Embedding dimension
    NUM_HEADS = 8  # Number of attention heads
    NUM_LAYERS = 4  # Number of transformer layers
    DFF = 512  # Hidden layer size in feed forward network
    DROPOUT_RATE = 0.1
    DENSE_UNITS = 64  # Units in final dense layer
    
    # Training parameters
    BATCH_SIZE = 1024  # Increased for faster training
    EPOCHS = 10  # Reduced for faster experimentation
    PATIENCE = 5  # Adjusted for fewer epochs
    LEARNING_RATE = 0.001  # Initial learning rate
    
    # Warmup steps for learning rate scheduler
    WARMUP_STEPS = 4000

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
