# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read in dataset 
df_sports_cognition = pd.read_csv("Data/Sport_Cognition.csv")

# # Display some data, statistics, and summary info
# print(df_sports_cognition.info())
# print(df_sports_cognition.describe())
# print("\n", df_sports_cognition.head())

# # Check for missing values
# print(df_sports_cognition.isnull().sum())

# Drop columns where more than the threshold of values are missing
threshold = 0.9
df_filtered = df_sports_cognition.dropna(thresh=len(df_sports_cognition) * (1 - threshold), axis=1)

# Remove rows where 'Finished' is False
df_filtered = df_filtered[df_filtered["Finished"] == "True"]

# Keep only rows where 'Q65' is "I am between the ages of 18-25."
df_filtered = df_filtered[df_filtered["Q65"] == "I am between the ages of 18-25."]

# Keep columns that start with "Q" or are "Duration (in seconds)"
df_filtered = df_filtered[[col for col in df_filtered.columns if col.startswith("Q")] + ["Duration (in seconds)"]]

# Drop the "Q_RecaptchaScore" column
df_filtered = df_filtered.drop(columns=["Q_RecaptchaScore"], errors="ignore")

# Display the new shape of the dataset
print("Original shape:", df_sports_cognition.shape)
print("New shape after dropping mostly empty columns:", df_filtered.shape)

# Save the cleaned dataset
df_filtered.to_csv("cleaned_SP_data.csv", index=False)