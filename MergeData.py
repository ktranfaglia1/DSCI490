import pandas as pd

# Read in data to merge
df1 = pd.read_csv("./Data/Survey_data.csv")
df2 = pd.read_csv("./Data/Standerized_SP_data.csv")

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

print (df1.shape)

# Export the dataset
df1.to_csv("./Data/Survey_data.csv")