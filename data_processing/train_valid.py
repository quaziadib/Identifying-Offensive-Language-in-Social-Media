import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv('dataset/OLID/olid-training-v1.0.tsv', sep="\t")
    df.drop(['subtask_b', 'subtask_c'], axis=1)
    X, y = df.tweet, df.subtask_a

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train.to_csv('dataset/OLID/train_valid_split/X_train.csv', index=False)
    X_valid.to_csv('dataset/OLID/train_valid_split/X_valid.csv', index=False)
    y_train.to_csv('dataset/OLID/train_valid_split/y_train.csv', index=False)
    y_valid.to_csv('dataset/OLID/train_valid_split/y_valid.csv', index=False)


if __name__ == '__main__':
    main()
