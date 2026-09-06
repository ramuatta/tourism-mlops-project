
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def main():

    # File paths
    input_path = "tourism_project/data/tourism.csv"
    output_dir = "tourism_project/data"

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    # Load dataset
    print("=" * 60)
    print("TOURISM DATA PREPARATION")
    print("=" * 60)

    df = pd.read_csv(input_path)

    print("\nOriginal Dataset Shape:", df.shape)
    print("\nOriginal Columns:")
    print(df.columns.tolist())

    # Remove unnecessary columns
    columns_to_remove = ["Unnamed: 0", "CustomerID"]

    existing_columns_to_remove = [
        col for col in columns_to_remove if col in df.columns
    ]

    df = df.drop(columns=existing_columns_to_remove)

    print("\nRemoved Columns:", existing_columns_to_remove)
    print("Dataset Shape After Removing Unnecessary Columns:", df.shape)

    # Target column
    target_column = "ProdTaken"

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    print("\nFeature Shape:", X.shape)
    print("Target Shape:", y.shape)

    print("\nTarget Distribution:")
    print(y.value_counts())
    print("\nTarget Distribution (%):")
    print((y.value_counts(normalize=True) * 100).round(2))

    # Train-test split
    # Stratify is used because ProdTaken is imbalanced
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Combine features and target before saving
    train_df = X_train.copy()
    train_df[target_column] = y_train

    test_df = X_test.copy()
    test_df[target_column] = y_test

    # Save prepared datasets
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nTrain Dataset Shape:", train_df.shape)
    print("Test Dataset Shape:", test_df.shape)

    print("\nTrain Target Distribution:")
    print(train_df[target_column].value_counts())

    print("\nTest Target Distribution:")
    print(test_df[target_column].value_counts())

    print("\nFiles saved successfully:")
    print(train_path)
    print(test_path)


if __name__ == "__main__":
    main()
