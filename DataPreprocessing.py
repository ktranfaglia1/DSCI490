import pandas as pd

df_filtered = pd.read_csv("./Data/FilteredData.csv")
surveyData = pd.read_csv("./Data/Survey_data.csv")

# print(df_filtered[df_filtered != surveyData])
removed_columns = set(df_filtered.columns) - set(surveyData.columns)
added_columns = set(surveyData.columns) - set(df_filtered.columns)


print("Removed columns:", removed_columns)
print("Removed columns:", added_columns)

print(df_filtered.shape)
print(surveyData.shape)
