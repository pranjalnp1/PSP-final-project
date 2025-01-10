# The Current State of the New York Tech Job market.
This repositry contains the code for visualisations for a report written analysisng the current state of the new york tech job market to gain insights into how aspiring new entrants can increase there chances of gaining job as competition grows. 

## 1. Overview  
This repository contains the code used for the analysis of exploring the relationship between COVID-19 vaccinations and case numbers over time. By analysing temporal patterns and performing lag-based correlations, this project seeks to understand whether vaccination efforts have had a delayed impact on reducing COVID-19 cases.  

The hypothesis explored in this analysis is:  
> "Vaccination efforts have a delayed effect on reducing COVID-19 cases, with variations based on regional factors"  

## 2. Project Structure  
The repository is organized as follows:  

- **`.circleci/`**: Configuration files for CircleCI integration to automate testing and ensure code reliability.  
- **`venv`**: Virtual environment where all dependencies are installed  
- **`main.py`**: Contains all of the code for visualisations.
- **`tests.py`**: Test scripts to ensure correctness of data functions and models.  
 **`.nycJobs.csv`**: Dataset for the analysis  
- **`requirements.txt`**: Specifies Python dependencies required for the project.  
- **`README.md`**: Project documentation.  

## 3. Dataset  
The dataset nycJobs.csv was sourced from the US government site "data.gov".


## 4. How to Run  
To run the analysis and generate visualizations locally:  

1. Clone the repository:  
    ```bash
    git clone https://github.com/pranjalnp1/PSP-final-project.git

2. Install the required dependencies from 'requirements.txt':
    ```bash
    pip install -r requirements.txt

4. Run the data analysis script - 'data_analysis.py':
   ```bash
   python main.py

6. To execute the test suite:
    ```bash
    python -m unittest tests.py

7. View the visualisations:

   visualisations will show up in pop ups and you can X out of them to view the next one.