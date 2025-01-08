import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from main import clean_nyc_jobs_data

class TestNYCJobsDataCleaning(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame with test data
        self.test_data = pd.DataFrame({
            'Job ID': [1, 2, 2, 3, 4],  # Note: ID 2 is duplicated
            'Agency': ['DEPT A', ' DEPT B ', None, 'DEPT D', 'DEPT E'],
            'Business Title': ['Title 1', 'Title 2', 'Title 3', None, 'Title 5'],
            'Salary Range From': [50000, 60000, np.nan, 70000, -1000],
            'Salary Range To': [75000, 55000, 80000, 90000, 80000],
            'Posting Date': ['2023-01-01', '2023-02-01', '2023-03-01', 'invalid_date', '2023-05-01']
        })
        
    def test_remove_duplicates(self):
        cleaned_df = clean_nyc_jobs_data(self.test_data)
        self.assertEqual(len(cleaned_df), len(cleaned_df['job_id'].unique()),
                        "Duplicates should be removed")

    def test_handle_missing_values(self):
        cleaned_df = clean_nyc_jobs_data(self.test_data)
        essential_cols = ['job_id', 'agency', 'business_title', 'salary_range_from', 'salary_range_to']
        for col in essential_cols:
            self.assertEqual(cleaned_df[col].isnull().sum(), 0,
                           f"No missing values should exist in {col}")

    def test_salary_range_validation(self):
        cleaned_df = clean_nyc_jobs_data(self.test_data)
        self.assertTrue(all(cleaned_df['salary_range_from'] <= cleaned_df['salary_range_to']),
                       "Salary range 'from' should not be greater than 'to'")

    def test_column_names_standardization(self):
        cleaned_df = clean_nyc_jobs_data(self.test_data)
        for col in cleaned_df.columns:
            self.assertTrue(col.islower() and ' ' not in col,
                          f"Column name {col} should be lowercase with underscores")

    def test_text_columns_stripped(self):
        cleaned_df = clean_nyc_jobs_data(self.test_data)
        self.assertTrue(all(cleaned_df['agency'].str.strip() == cleaned_df['agency']),
                       "Text columns should be stripped of leading/trailing spaces")

    def test_date_conversion(self):
        cleaned_df = clean_nyc_jobs_data(self.test_data)
        self.assertTrue(pd.api.types.is_datetime64_dtype(cleaned_df['posting_date']),
                       "Date columns should be converted to datetime")

# Run the tests
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)