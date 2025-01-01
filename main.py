
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kagglehub



df = pd.read_csv("global_tech_salary.txt")

# Clean and format the data
# Remove leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Remove leading and trailing spaces from string values in the dataframe
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Convert columns to appropriate data types
df['work_year'] = df['work_year'].astype(int)
df['salary'] = df['salary'].astype(float)
df['salary_in_usd'] = df['salary_in_usd'].astype(float)
df['remote_ratio'] = df['remote_ratio'].astype(int)

# Display the cleaned and formatted dataframe
print("\nCleaned and Formatted DataFrame:")
print(df.head())

# Save the cleaned and formatted dataframe to a new CSV file
cleaned_file_path = 'cleaned_tech_jobs.csv'
df.to_csv(cleaned_file_path, index=False)

print(f"\nCleaned and formatted data has been saved to {cleaned_file_path}.")









