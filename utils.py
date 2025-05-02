import pandas as pd


def sample_per_class(df, label_columns):

    sampled_data = []
    for label in label_columns:
        for class_value in df[label].unique():
            subset = df[df[label] == class_value]
            if len(subset) >= 10:
                sampled_subset = subset.sample(n=10, random_state=42)
            else:
                sampled_subset = subset
            sampled_data.append(sampled_subset)
    return pd.concat(sampled_data).drop_duplicates().reset_index(drop=True)
