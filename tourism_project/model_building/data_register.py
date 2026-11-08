import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"  # path to the raw tourism.csv file inside the data folder

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# Flag any extra columns (e.g. a saved pandas index) so they can be removed in data preparation
extra = [c for c in df.columns if c not in expected_columns]

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Unexpected extra columns:", extra)
print("Total missing values:", int(df.isnull().sum().sum()))
print("ProdTaken distribution:")
print(df["ProdTaken"].value_counts())
