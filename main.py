import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import json

# Set your Kaggle API token directly
os.environ['KAGGLE_API_TOKEN'] = "KGAT_fbfe0f01e1bc05ac0caaba68d5bc4436"

# Download the data
path = kagglehub.competition_download('ga-customer-revenue-prediction')
print("Path to competition files:", path)

# Read the data (ensure 'train.csv' matches the exact file name in your folder)
file_path = f"{path}\\train.csv"
df = pd.read_csv(file_path)

# Display the first 5 rows of the dataset
print(df.head())

df.shape
df.head()
df.info()

df.columns.tolist()

#function
def parse_json_column(df, column):
    parsed = df[column].apply(json.loads)
    parsed_df = pd.json_normalize(parsed)

    parsed_df.columns = [
        f"{column}_{col}" for col in parsed_df.columns
    ]

    return parsed_df

#Clean Dataset Format
traffic = parse_json_column(df, "trafficSource")
device = parse_json_column(df, "device")
totals = parse_json_column(df, "totals")

df_clean = pd.concat(
    [
        df.drop(columns=["trafficSource", "device", "totals"]),
        traffic,
        device,
        totals
    ],
    axis=1
)

df_clean.head()
df_clean.columns.tolist()

# Data Cleaning
## duplicates
df_clean.duplicated().sum()

df_clean.info()

## check unique sess
df_clean[
    ["fullVisitorId", "visitId"]
].duplicated().sum()

## missing values
missing = (
    df_clean.isna()
    .mean()
    .sort_values(ascending=False)
)

missing.head(20)

# Show all columns (variables) without cropping
# 1. Set the global option using the 'pd' module
pd.set_option('display.max_columns', None)
# 2. Now look at your DataFrame normally
df_clean.head()

#wanna see all variables
for col in df_clean.columns:
    print(col)

# Option 2: A simple Python list format
print(list(df_clean.columns))