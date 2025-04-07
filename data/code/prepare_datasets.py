import sys
import os
sys.path.append(os.path.abspath("."))  
from data.code.loader import CICIDataLoader

if __name__ == "__main__":
    loader = CICIDataLoader() 
    loader.run_full_preparation()
