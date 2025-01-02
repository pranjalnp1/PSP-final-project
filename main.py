
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kagglehub


# reading datset
df = pd.read_csv("nycJobs.csv")



#Adding NaNs to empty cells.
def replaceEmptyCellsWithNaN(df):
    #replacing empty cells with NaN
    df.replace('', np.nan, inplace=True)


#function that formats cells that are all capitalized to be more readable 
def formatCapitalizedCells(cell):
    # Check if the value is a string and is in all uppercase
    if isinstance(cell, str) and cell.isupper():
        # Convert to lowercase and capitalize the first letter
        return cell.capitalize()
    return cell

replaceEmptyCellsWithNaN(df)
df = df.applymap(formatCapitalizedCells)





print(df["Civil Service Title"])








