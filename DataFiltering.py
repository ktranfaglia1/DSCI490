import pandas as pd

# Read in dataset 
df_sports_cognition = pd.read_csv("Data/Survey_data.csv")

# # Display some data, statistics, and summary info
# print(df_sports_cognition.info())
# print(df_sports_cognition.describe())
# print("\n", df_sports_cognition.head())

# # Check for missing values
# print(df_sports_cognition.isnull().sum())

# # Drop columns where more than the threshold of values are missing
# threshold = 0.9
# df_filtered = df_sports_cognition.dropna(thresh=len(df_sports_cognition) * (1 - threshold), axis=1)

# # Remove rows where 'Finished' is False
# df_filtered = df_filtered[df_filtered["Finished"] == "True"]

# # Keep only rows where 'Q65' is "I am between the ages of 18-25."
# df_filtered = df_filtered[df_filtered["Q65"] == "I am between the ages of 18-25."]

# # Keep columns that start with "Q" or are "Duration (in seconds)"
# df_filtered = df_filtered[[col for col in df_filtered.columns if col.startswith("Q")] + ["Duration (in seconds)"]]

# # Drop the "Q_RecaptchaScore" column
# df_filtered = df_filtered.drop(columns=["Q_RecaptchaScore"], errors="ignore")

df_filtered = df_sports_cognition

# Define the conditions for each attention check
condition_QID91 = df_filtered["QID91"] == "True"
condition_Q89 = df_filtered["Q89"] == "True"
condition_Q90 = df_filtered["Q90"] == "75 degrees or higher"

# Count how many conditions each row meets
df_filtered["attention_pass_count"] = condition_QID91 + condition_Q89 + condition_Q90

# Keep only rows where at least 2 out of 3 checks pass
df_filtered = df_filtered[df_filtered["attention_pass_count"] >= 2]



# Display the new shape of the dataset
print("Original shape:", df_sports_cognition.shape)
print("New shape after dropping mostly empty columns:", df_filtered.shape)

# Save the cleaned dataset
df_filtered.to_csv("Survey_data.csv", index=False)