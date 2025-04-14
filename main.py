import pandas as pd
from utils import sample_per_class
import argparse

from experiments_runners.TFTransformerExperimentRunner import TFTransformerExperimentRunner
from experiments_runners.LSTMExperimentRunner import LSTMExperimentRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to use", default="TabTransformer")
    args = parser.parse_args()
    print(f"model selected: {args.model}")

    train_df = pd.read_csv("data/concatenated/train.csv")
    test_df = pd.read_csv("data/concatenated/test.csv")

    labels = ['label_2', 'label_6', 'label_19']
    # Apply separately
    train_df = sample_per_class(train_df, labels)
    test_df = sample_per_class(test_df, labels)

    print('loading data done!')

    if args.model == "LSTM":
        runner = LSTMExperimentRunner(train_df, test_df)
        runner.run_experiments()
        print("=== All experiments completed successfully ===")
    elif args.model == "TabTransformer":
        runner = TFTransformerExperimentRunner(train_df, test_df)
        runner.run_experiments()
        print("=== All experiments completed successfully ===")
    else:
        print("model not recognized")



if __name__ == "__main__":
    main()
