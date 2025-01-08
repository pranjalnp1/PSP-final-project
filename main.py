import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from io import StringIO


filepath = "nycJobs.csv"
df = pd.read_csv(filepath)


# Function to clean the dataset
def clean_nyc_jobs_data(df):
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Handle missing values
    # Drop rows where essential columns have missing values
    essential_columns = ['Job ID', 'Agency', 'Business Title', 'Salary Range From', 'Salary Range To']
    df = df.dropna(subset=essential_columns)

    # Standardize column names (strip spaces and convert to lowercase)
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # Convert salary columns to numeric
    df['salary_range_from'] = pd.to_numeric(df['salary_range_from'], errors='coerce')
    df['salary_range_to'] = pd.to_numeric(df['salary_range_to'], errors='coerce')

    # Remove rows with invalid salary ranges (e.g., from > to)
    df = df[df['salary_range_from'] <= df['salary_range_to']]

    # Standardize text columns (strip extra spaces)
    text_columns = ['agency', 'business_title', 'civil_service_title', 'job_category']
    for col in text_columns:
        if col in df.columns:  # Check if column exists
            df[col] = df[col].str.strip()

    # Convert date columns to datetime format
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
    monthly_distribution = cleaned_df.groupby(cleaned_df['posting_date'].dt.month).size()
    print("\
    Postings by month (all years combined):")
    print(monthly_distribution.to_string())




def current_tech_positions(df):
    # Filter for tech-related jobs based on keywords in 'business_title' and 'job_description'
    tech_keywords = [ 'software', 'developer', 'programming', 'coder', 'full stack', 'frontend', 'backend',
                'web developer', 'application developer' 'data scientist', 'data analyst', 'machine learning', 'ai ', 'artificial intelligence',
                'analytics', 'big data', 'data engineer', 'statistics', 'statistical''security', 'cyber', 'information security', 'infosec', 'network security',
                'security analyst', 'security engineer''it ', 'information technology', 'systems admin', 'network admin', 'database admin',
                'system engineer', 'infrastructure', 'devops', 'cloud' 'computer', 'technical', 'technology', 'sql', 'python', 'java', 'javascript',
                'analyst programmer', 'computer associate']

    # Combine title and description for filtering
    cleaned_df['combined_text'] = cleaned_df['business_title'].str.lower() + ' ' + cleaned_df['job_description'].str.lower()
    tech_jobs = cleaned_df[cleaned_df['combined_text'].str.contains('|'.join(tech_keywords), na=False)]

    # Group by career level and sum the number of positions
    tech_jobs_grouped = tech_jobs.groupby('career_level')['#_of_positions'].sum().reset_index()


    # Count the number of job listings for each career level
    job_counts = tech_jobs['career_level'].value_counts().reset_index()
    job_counts.columns = ['career_level', 'job_listings']

    # Print the number of job listings for each career level
    print("Number of job listings for each career level:")
    print(job_counts)

    # Print relevant statistics for the number of positions
    print("\
    Relevant statistics for tech job listings by career level (total positions):")
    print(tech_jobs_grouped)

    # Plot the results
    plt.figure(figsize=(10, 6))
    sns.barplot(data=tech_jobs_grouped, x='career_level', y='#_of_positions', palette='viridis')
    plt.title('Number of Tech Job Listings by Career Level', fontsize=16)
    plt.xlabel('Career Level', fontsize=12)
    plt.ylabel('Number of Listings', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

current_tech_positions(cleaned_df)


def categorise_entry_level_roles(df):
    # Define disciplines and their associated keywords with expanded terms
    disciplines = {
        'Software Engineering': [
            'software engineer', 'developer', 'programmer', 'full stack', 'frontend', 'backend', 
            'web developer', 'mobile developer', 'java developer', 'python developer', 'coding'
        ],
        'Data Science & Analytics': [
            'data scientist', 'data analyst', 'machine learning', 'data engineer', 'analytics',
            'business intelligence', 'bi developer', 'statistical', 'data visualization'
        ],
        'Cybersecurity': [
            'security engineer', 'cyber', 'information security', 'security analyst', 
            'cybersecurity', 'cyber security', 'information assurance', 'security operations'
        ],
        'IT & Systems': [
            'systems admin', 'network admin', 'database admin', 'system engineer', 
            'infrastructure', 'devops', 'cloud engineer', 'it support', 'help desk',
            'technical support', 'systems analyst', 'network engineer'
        ],
        'Solutions Architecture': [
            'solutions architect', 'cloud architect', 'enterprise architect', 
            'technical architect', 'systems architect'
        ],
        'QA & Testing': [
            'quality assurance', 'qa engineer', 'test engineer', 'software tester',
            'quality engineer', 'automation engineer'
        ],
        'Other Tech': [
            'technology specialist', 'computer operator', 'computer associate',
            'digital', 'tech support', 'technical assistant'
        ]
    }

    # Function to assign a discipline based on title and description
    def assign_discipline(title, description):
        title = str(title).lower()
        description = str(description).lower()
        
        for discipline, keywords in disciplines.items():
            if any(keyword in title or keyword in description for keyword in keywords):
                return discipline
        return None  # Return None if no tech discipline matches

    # Filter for entry level and internship roles
    entry_keywords = ['entry level', 'entry-level', 'intern', 'internship', 'junior', 'associate', 'trainee']
    entry_roles = df[
        df.apply(lambda x: any(keyword in str(x['business_title']).lower() or 
                              keyword in str(x['job_description']).lower() 
                              for keyword in entry_keywords), axis=1)
    ]

    # Assign disciplines to entry level roles
    entry_roles['Discipline'] = entry_roles.apply(
        lambda x: assign_discipline(x['business_title'], x['job_description']), axis=1
    )

    # Remove non-tech roles (where Discipline is None)
    entry_roles = entry_roles[entry_roles['Discipline'].notna()]

    # Get the distribution of roles by discipline
    discipline_distribution = entry_roles['Discipline'].value_counts()

    # Create a more detailed DataFrame with percentages
    analysis_df = pd.DataFrame({
        'Count': discipline_distribution,
        'Percentage': (discipline_distribution / discipline_distribution.sum() * 100).round(1)
    })

    # Plot the distribution
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(discipline_distribution)), discipline_distribution, color='skyblue')
    plt.title('Distribution of Entry-Level & Internship Tech Roles by Discipline', fontsize=14, pad=20)
    plt.xlabel('Discipline', fontsize=12)
    plt.ylabel('Number of Roles', fontsize=12)
    plt.xticks(range(len(discipline_distribution)), discipline_distribution.index, rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

    return analysis_df

# Run the function and display the results
entry_level_analysis = categorise_entry_level_roles(cleaned_df)
print("\
Detailed Analysis of Entry-Level & Internship Tech Roles:")
print(entry_level_analysis.to_string())