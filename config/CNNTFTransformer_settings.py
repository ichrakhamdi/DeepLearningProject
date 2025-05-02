import warnings
from pathlib import Path
import pandas as pd

from config.TFTrasformer_settings import TFTransformerSettings

class CNNTFTransformerSettings(TFTransformerSettings):
    """
    Settings for CNN+Transformer hybrid; inherits TFTransformerSettings and adds CNN parameters.
    """
    # CNN front-end parameters
    CNN_KERNEL_SIZE = 3
    CNN_PADDING = 1

    def __init__(self):
        super().__init__() 