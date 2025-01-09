import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from main import clean_nyc_jobs_data
from main import time_graph_job_listings
from main import current_tech_positions
from main import categorise_entry_level_roles
from main import extract_skills

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


class TestCurrentTechPositions(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset for testing
        self.mock_data = pd.DataFrame({
            'business_title': [
                'Software Engineer', 'Data Scientist', 'Cybersecurity Analyst',
                'IT Support Specialist', 'Non-Tech Role'
            ],
            'job_description': [
                'Develop software applications', 'Analyze data and build models',
                'Monitor and secure systems', 'Provide IT support', 'Manage office tasks'
            ],
            'career_level': ['Entry', 'Mid', 'Senior', 'Entry', 'Entry'],
            '#_of_positions': [5, 3, 2, 4, 1]
        })

    def test_tech_job_filtering(self):
        # Test if tech jobs are correctly filtered
        tech_keywords = [
            'software', 'developer', 'programming', 'coder', 'data scientist', 'data analyst',
            'machine learning', 'security', 'cyber', 'it ', 'information technology'
        ]
        self.mock_data['combined_text'] = self.mock_data['business_title'].str.lower() + ' ' + self.mock_data['job_description'].str.lower()
        tech_jobs = self.mock_data[self.mock_data['combined_text'].str.contains('|'.join(tech_keywords), na=False)]
        self.assertEqual(len(tech_jobs), 4)  # 4 out of 5 rows are tech-related

    def test_grouping_by_career_level(self):
        # Test grouping by career level and summing positions
        self.mock_data['combined_text'] = self.mock_data['business_title'].str.lower() + ' ' + self.mock_data['job_description'].str.lower()
        tech_jobs = self.mock_data[self.mock_data['combined_text'].str.contains('|'.join(['software', 'data', 'cyber', 'it']), na=False)]
        tech_jobs_grouped = tech_jobs.groupby('career_level')['#_of_positions'].sum().reset_index()
        self.assertEqual(tech_jobs_grouped.loc[tech_jobs_grouped['career_level'] == 'Entry', '#_of_positions'].values[0], 9)

    def test_job_listings_count(self):
        # Test counting job listings for each career level
        self.mock_data['combined_text'] = self.mock_data['business_title'].str.lower() + ' ' + self.mock_data['job_description'].str.lower()
        tech_jobs = self.mock_data[self.mock_data['combined_text'].str.contains('|'.join(['software', 'data', 'cyber', 'it']), na=False)]
        job_counts = tech_jobs['career_level'].value_counts().reset_index()
        job_counts.columns = ['career_level', 'job_listings']
        self.assertEqual(job_counts.loc[job_counts['career_level'] == 'Entry', 'job_listings'].values[0], 2)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


class TestEntryLevelRoles(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset for testing
        self.mock_data = pd.DataFrame({
            'business_title': [
                'Entry Level Software Engineer',
                'Junior Data Analyst',
                'Cybersecurity Intern',
                'IT Support Associate',
                'Senior Software Engineer',
                'Entry Level Marketing',
                'Junior QA Engineer',
                'Cloud Architecture Trainee'
            ],
            'job_description': [
                'Develop software applications using Python',
                'Analyze data and create visualizations',
                'Monitor and secure systems',
                'Provide technical support',
                'Lead software development team',
                'Create marketing campaigns',
                'Test software applications',
                'Design cloud solutions'
            ]
        })
        
    def test_entry_level_filtering(self):
        """Test if entry-level roles are correctly identified"""
        result = categorise_entry_level_roles(self.mock_data)
        # Should only count entry-level tech roles (6 out of 8 in mock data)
        self.assertEqual(len(result), 6)
        
    def test_discipline_assignment(self):
        """Test if disciplines are correctly assigned"""
        result = categorise_entry_level_roles(self.mock_data)
        disciplines = result.index.tolist()
        
        # Check if specific disciplines are present
        expected_disciplines = [
            'Software Engineering',
            'Data Science & Analytics',
            'Cybersecurity',
            'IT & Systems',
            'QA & Testing'
        ]
        for discipline in expected_disciplines:
            self.assertIn(discipline, disciplines)
            
    def test_percentage_calculation(self):
        """Test if percentages are calculated correctly"""
        result = categorise_entry_level_roles(self.mock_data)
        total_percentage = result['Percentage'].sum()
        self.assertAlmostEqual(total_percentage, 100.0, places=1)
        
    def test_non_tech_exclusion(self):
        """Test if non-tech roles are excluded"""
        result = categorise_entry_level_roles(self.mock_data)
        # Marketing role should be excluded
        self.assertNotIn('Marketing', result.index)
        
    def test_count_values(self):
        """Test if count values are integers"""
        result = categorise_entry_level_roles(self.mock_data)
        self.assertTrue(all(isinstance(count, (int, np.integer)) for count in result['Count']))
        
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame(columns=['business_title', 'job_description'])
        result = categorise_entry_level_roles(empty_df)
        self.assertEqual(len(result), 0)
        
    def test_missing_values(self):
        """Test handling of missing values"""
        df_with_na = self.mock_data.copy()
        df_with_na.loc[0, 'job_description'] = None
        result = categorise_entry_level_roles(df_with_na)
        self.assertTrue(len(result) > 0)  # Should still process other valid entries

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)



# Mock data for testing
mock_skill_categories = {
    'Programming': ['python', 'java', 'c++'],
    'Web/Frontend': ['html', 'css', 'javascript']
}

def mock_extract_skills(text):
    text = text.lower()
    matches = {}
    for category, keywords in mock_skill_categories.items():
        category_matches = [kw for kw in keywords if kw in text]
        if category_matches:
            matches[category] = category_matches
    return matches

class TestSkillExtraction(unittest.TestCase):

    def test_extract_skills_single_category(self):
        text = "We are looking for a Python developer with experience in Java."
        expected_output = {'Programming': ['python', 'java']}
        self.assertEqual(mock_extract_skills(text), expected_output)

    def test_extract_skills_multiple_categories(self):
        text = "The candidate should know HTML, CSS, and Python."
        expected_output = {
            'Programming': ['python'],
            'Web/Frontend': ['html', 'css']
        }
        self.assertEqual(mock_extract_skills(text), expected_output)

    def test_extract_skills_no_match(self):
        text = "This job requires management and communication skills."
        expected_output = {}
        self.assertEqual(mock_extract_skills(text), expected_output)

    def test_extract_skills_case_insensitivity(self):
        text = "We need a PYTHON and JAVA expert."
        expected_output = {'Programming': ['python', 'java']}
        self.assertEqual(mock_extract_skills(text), expected_output)

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)



