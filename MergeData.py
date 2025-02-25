#  Author: Kyle Tranfaglia
#  Title: DSCI490 - MergeData
#  Last updated:  02/24/25
#  Description: This program is a utility program to merge columns of one dataset to another
import pandas as pd

# Read in data to merge
df1 = pd.read_csv("./Data/Survey_data.csv")
df2 = pd.read_csv("./Data/Survey_data_test.csv")

print (df1.shape)

# Replace df1 columns with df2 columns
df1["Q2"] = df2["Q2"]
df1["Q3"] = df2["Q3 (Minutes)"]
df1["Q4"] = df2["Q4"]
df1["Q5"] = df2["Q5 (Hours)"]

print (df1["Q2"])
print (df1["Q3"])
print (df1["Q4"])
print (df1["Q5"])

df1['Attention_score'] = df2['attention_pass_count']

print (df1.shape)

# Keep columns that meet the criteria
df1 = df1[[col for col in df1.columns if col.startswith("Q")] + ["Duration (in seconds)"] + ["Attention_score"]]

# Export the dataset
df1.to_csv("./Data/Survey_data.csv")