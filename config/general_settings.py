class GeneralSettings:
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
    CLASS_TYPES = ['6class', 'binary', '19class']
    AUGMENTATIONS = [True, False]
    CLASS_WEIGHTS = [True, False]
    WINDOW_SIZE = 4

    # Model configuration
    LEARNING_RATE = 0.001
    BATCH_SIZE = 1024
    EPOCHS = 50
    PATIENCE = 5
    WEIGHT_DECAY = 0.0001

    RESULTS_PATH = None
    ENCODERS_PATH = None
    SCALERS_PATH = None
