import pandas as pd
import numpy as np
import re
from io import StringIO


filepath = "nycJobs.csv"
df = pd.read_csv(filepath)


# Function to clean the dataset
# Adjusting the cleaning function to handle missing columns gracefully
def clean_nyc_jobs_data(df):
    # 1. Remove duplicate rows
    df = df.drop_duplicates()

    # 2. Handle missing values
    # Drop rows where essential columns have missing values
    essential_columns = ['Job ID', 'Agency', 'Business Title', 'Salary Range From', 'Salary Range To']
    df = df.dropna(subset=essential_columns)

    # 3. Standardize column names (strip spaces and convert to lowercase)
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # 4. Convert salary columns to numeric
    df['salary_range_from'] = pd.to_numeric(df['salary_range_from'], errors='coerce')
    df['salary_range_to'] = pd.to_numeric(df['salary_range_to'], errors='coerce')

    # 5. Remove rows with invalid salary ranges (e.g., from > to)
    df = df[df['salary_range_from'] <= df['salary_range_to']]

    # 6. Standardize text columns (strip extra spaces)
    text_columns = ['agency', 'business_title', 'civil_service_title', 'job_category']
    for col in text_columns:
        if col in df.columns:  # Check if column exists
            df[col] = df[col].str.strip()

    # 7. Convert date columns to datetime format
    date_columns = ['posting_date', 'post_until', 'posting_updated', 'process_date']
    for col in date_columns:
        if col in df.columns:  # Check if column exists
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


# Clean the dataset
cleaned_df = clean_nyc_jobs_data(df)

# Display the first few rows of the cleaned dataset
print(cleaned_df.head())
print(cleaned_df.info())
