#  Author: Kyle Tranfaglia
#  Title: DSCI490 - TheDataFixer
#  Last updated:  02/24/25
#  Description: This program is a utility program to add necessary columns to a dataset
import pandas as pd

# Load dataset 1 and dataset 2
df1 = pd.read_csv("./Data/Survey_data.csv")
df2 = pd.read_csv("./Data/SP_test.csv")

# Display or save the updated dataset
print(df1.head())

# Remove 'Q78' from df1
df1.drop(columns=['Q78'], inplace=True)

# Select columns to add from df2
columns_to_add = ['Q78', 'Q78_1_TEXT']

# Concatenate along columns (assuming df1 and df2 have the same row order)
df1 = pd.concat([df1, df2[columns_to_add]], axis=1)

# Display or save the updated dataset
print(df1.head())

# Save to a new file
df1.to_csv("./Data/The_survey_data.csv", index=False)
