
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kagglehub



df = pd.read_csv("global_tech_salary.txt")


def cleanData(df):
            df.columns = df.columns.str.strip()
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            df['work_year'] = df['work_year'].astype(int)
            df['salary'] = df['salary'].astype(float)
            df['salary_in_usd'] = df['salary_in_usd'].astype(float)
            df['remote_ratio'] = df['remote_ratio'].astype(int)
            print(df)
            return df


cleanData(df)






