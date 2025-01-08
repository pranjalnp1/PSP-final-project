import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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





class TestTimeGraphJobListings(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset for testing
        self.mock_data = pd.DataFrame({
            'posting_date': pd.date_range(start='2021-01-01', periods=100, freq='D')
        })
        # Add some NaN values to test error handling
        self.mock_data.loc[10:15, 'posting_date'] = np.nan
        # Convert posting date to datetime
        self.mock_data['posting_date'] = pd.to_datetime(self.mock_data['posting_date'], errors='coerce')

    def test_date_conversion(self):
        # Test if dates are correctly converted to datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.mock_data['posting_date']))
        # Test if NaN values are handled correctly
        self.assertEqual(self.mock_data['posting_date'].isna().sum(), 6)

    def test_date_range(self):
        # Test if the date range is calculated correctly
        earliest_date = self.mock_data['posting_date'].min()
        latest_date = self.mock_data['posting_date'].max()
        expected_range = pd.Timestamp('2021-04-10') - pd.Timestamp('2021-01-01')
        actual_range = latest_date - earliest_date
        self.assertEqual(actual_range.days, expected_range.days)

    def test_monthly_post_counts(self):
        # Test if monthly post counts are calculated correctly
        monthly_posts = self.mock_data.groupby(self.mock_data['posting_date'].dt.to_period('M')).size()
        # January should have 25 posts (31 days - 6 NaN values)
        self.assertEqual(monthly_posts.loc['2021-01'], 25)
        # Check total number of months
        self.assertEqual(len(monthly_posts), 4)  # Jan, Feb, Mar, Apr

    def test_yearly_post_counts(self):
        # Test if yearly post counts are calculated correctly
        yearly_posts = self.mock_data.groupby(self.mock_data['posting_date'].dt.year).size()
        # Should have 94 posts in 2021 (100 total - 6 NaN)
        self.assertEqual(yearly_posts[2021], 94)
        # Should only have data for 2021
        self.assertEqual(len(yearly_posts), 1)

    def test_plot_attributes(self):
        # Test if the plot is generated with correct attributes
        monthly_posts = self.mock_data.groupby(self.mock_data['posting_date'].dt.to_period('M')).size()
        fig, ax = plt.subplots(figsize=(15, 6))
        monthly_posts.plot(kind='line', marker='o', ax=ax)
        ax.set_title('Number of Job Postings Over Time', fontsize=15)
        ax.set_xlabel('Date', fontsize=15)
        ax.set_ylabel('Number of Postings', fontsize=15)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Test figure size
        self.assertEqual(fig.get_size_inches().tolist(), [15, 6])
        
        # Test plot title and labels
        self.assertEqual(ax.get_title(), 'Number of Job Postings Over Time')
        self.assertEqual(ax.get_xlabel(), 'Date')
        self.assertEqual(ax.get_ylabel(), 'Number of Postings')
        
        # Test if grid is enabled
        self.assertTrue(ax.get_xgridlines()[0].get_visible())
        
        plt.close()  # Close the plot after testing

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
