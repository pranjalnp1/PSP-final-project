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




class TestTimeGraphJobListings(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset for testing
        self.mock_data = pd.DataFrame({
            'posting_date': pd.date_range(start='2021-01-01', periods=100, freq='D')
        })
        # Add some NaN values to test error handling
        self.mock_data.loc[10:15, 'posting_date'] = np.nan

    def test_date_conversion(self):
        # Test if dates are correctly converted
        self.mock_data['posting_date'] = pd.to_datetime(self.mock_data['posting_date'], errors='coerce')
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.mock_data['posting_date']))

    def test_date_range(self):
        # Test if the date range is calculated correctly
        earliest_date = self.mock_data['posting_date'].min()
        latest_date = self.mock_data['posting_date'].max()
        self.assertEqual(earliest_date, pd.Timestamp('2021-01-01'))
        self.assertEqual(latest_date, pd.Timestamp('2021-04-10'))

    def test_monthly_post_counts(self):
        # Test if monthly post counts are calculated correctly
        monthly_posts = self.mock_data.groupby(self.mock_data['posting_date'].dt.to_period('M')).size()
        self.assertEqual(len(monthly_posts), 4)  # Jan, Feb, Mar, Apr
        self.assertEqual(monthly_posts.loc['2021-01'], 31)  # 31 days in Jan

    def test_yearly_post_counts(self):
        # Test if yearly post counts are calculated correctly
        yearly_posts = self.mock_data.groupby(self.mock_data['posting_date'].dt.year).size()
        self.assertEqual(yearly_posts[2021], 100 - 6)  # 100 total rows - 6 NaN rows

    def test_plot_generation(self):
        # Test if the plot is generated without errors
        try:
            monthly_posts = self.mock_data.groupby(self.mock_data['posting_date'].dt.to_period('M')).size()
            plt.figure(figsize=(15, 6))
            monthly_posts.plot(kind='line', marker='o')
            plt.title('Number of Job Postings Over Time', fontsize=15)
            plt.xlabel('Date', fontsize=15)
            plt.ylabel('Number of Postings', fontsize=15)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.close()  # Close the plot after testing
        except Exception as e:
            self.fail(f"Plot generation failed with error: {e}")

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
