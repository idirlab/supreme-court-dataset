import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('clean_data_with_details.csv')

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nBasic Info:")
print(df.info())

print("\nDescriptive Statistics for Numerical Columns:")
print(df.describe())

# Count non-null values for each column
print("\nNon-null counts:")
print(df.count())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Show percentage of missing values
print("\nPercentage of missing values:")
print((df.isnull().sum() / len(df)) * 100)

# Display some sample rows
print("\nFirst 5 rows:")
print(df.head())

# Text column statistics (if they exist)
text_columns = ['facts', 'question', 'conclusion', 'name']
existing_text_cols = [col for col in text_columns if col in df.columns]

if existing_text_cols:
    print("\nText column statistics:")
    for col in existing_text_cols:
        if df[col].dtype == 'object':
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                lengths = non_null_values.astype(str).str.len()
                print(f"\n{col}:")
                print(f"  - Non-null entries: {len(non_null_values)}")
                print(f"  - Average length: {lengths.mean():.2f} characters")
                print(f"  - Min length: {lengths.min()}")
                print(f"  - Max length: {lengths.max()}")

# Status distribution (if exists)
if 'status' in df.columns:
    print("\nStatus distribution:")
    print(df['status'].value_counts())

# Term distribution (if exists)
if 'term' in df.columns:
    print("\nTerm distribution:")
    print(df['term'].value_counts().sort_index())