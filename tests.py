import unittest
import pandas as pd
from io import StringIO

from main import cleanData


class TestCleanData(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = StringIO("""work_year,experience_level,employment_type,job_title,salary,salary_currency,salary_in_usd,employee_residence,remote_ratio,company_location,company_size
        2023, MI, FT, Data Analyst, 165000, USD, 165000, AU, 0, AU, M
        2023, MI, FT, Data Analyst, 70000, USD, 70000, US, 100, US, M
        2024, MI, FT, Machine Learning Engineer, 85000, EUR, 94444, IE, 100, IE, M
        2024, SE, FT, Data Scientist, 92700, USD, 92700, US, 0, US, M
        2023, MI, FT, Research Engineer, 150000, USD, 150000, US, 0, US, M
        """)
        self.df = pd.read_csv(self.data)

    def test_clean_data(self):
        # Function to clean and format the data
        def cleanData(df):
            df.columns = df.columns.str.strip()
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            df['work_year'] = df['work_year'].astype(int)
            df['salary'] = df['salary'].astype(float)
            df['salary_in_usd'] = df['salary_in_usd'].astype(float)
            df['remote_ratio'] = df['remote_ratio'].astype(int)
            return df

        cleaned_df = cleanData(self.df)

        # Check if column names are stripped
        self.assertTrue(all(cleaned_df.columns == ['work_year', 'experience_level', 'employment_type', 'job_title', 'salary', 'salary_currency', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']))

        # Check if string values are stripped
        self.assertTrue(all(cleaned_df['experience_level'] == ['MI', 'MI', 'MI', 'SE', 'MI']))

        # Check if data types are correct
        self.assertTrue(cleaned_df['work_year'].dtype == 'int')
        self.assertTrue(cleaned_df['salary'].dtype == 'float')
        self.assertTrue(cleaned_df['salary_in_usd'].dtype == 'float')
        self.assertTrue(cleaned_df['remote_ratio'].dtype == 'int')

if __name__ == '__main__':
    unittest.main()