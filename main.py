import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
#comment


def time_graph_job_listings():
    date_columns = [col for col in cleaned_df.columns if 'date' in col.lower()]


    # Convert posting date to datetime if it's not already
    cleaned_df['posting_date'] = pd.to_datetime(cleaned_df['posting_date'], errors='coerce')

    # Get the date range
    earliest_date = cleaned_df['posting_date'].min()
    latest_date = cleaned_df['posting_date'].max()
    date_range = latest_date - earliest_date

    # Count posts by month
    monthly_posts = cleaned_df.groupby(cleaned_df['posting_date'].dt.to_period('M')).size()

    # Create a line plot of postings over time
    plt.figure(figsize=(15, 6))
    monthly_posts.plot(kind='line', marker='o')
    plt.title('Number of Job Postings Over Time', fontsize = 15)
    plt.xlabel('Date', fontsize = 15)
    plt.ylabel('Number of Postings',  fontsize = 15)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7) 
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\
    Date Range Summary:")
    print(f"Earliest posting date: {earliest_date.strftime('%Y-%m-%d')}")
    print(f"Latest posting date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"Total time span: {date_range.days} days")

    # Show distribution by year
    yearly_posts = cleaned_df.groupby(cleaned_df['posting_date'].dt.year).size()
    print("\
    Postings by year:")
    print(yearly_posts.to_string())

    # Show distribution by month (across all years)
    monthly_distribution = cleaned_df.groupby(cleaned_df['Posting Date'].dt.month).size()
    print("\
    Postings by month (all years combined):")
    print(monthly_distribution.to_string())



time_graph_job_listings()