
import pandas as pd
import sys

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch"
]

def main():
    data_path = "tourism_project/data/tourism.csv"

    print("=" * 60)
    print("TOURISM DATA REGISTRATION")
    print("=" * 60)

    try:
        df = pd.read_csv(data_path)
        print("\nDataset loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)

    # Validate columns
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]

    print("\nExpected Columns:", len(EXPECTED_COLUMNS))
    print("Actual Columns:", len(df.columns))

    if missing_columns:
        print("\nMissing expected columns:")
        for col in missing_columns:
            print("-", col)
        print("\nDataset validation FAILED.")
        sys.exit(1)
    else:
        print("\nAll expected columns are present.")
        print("Dataset validation PASSED.")

    # Dataset summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"\nDataset Shape: {df.shape}")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")

    print("\nColumn Data Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nTarget Distribution - ProdTaken:")
    print(df["ProdTaken"].value_counts(dropna=False))

    print("\nData registration completed successfully.")


if __name__ == "__main__":
    main()
