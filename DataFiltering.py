import pandas as pd

# Read in dataset 
df_sports_cognition = pd.read_csv("Data/Survey_data.csv")

# Display original shape
print("Original shape:", df_sports_cognition.shape)

# Display some data, statistics, and summary info
print(df_sports_cognition.info())
print(df_sports_cognition.describe())
print("\n", df_sports_cognition.head())

# Check for missing values
print(df_sports_cognition.isnull().sum())

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

df_filtered = df_sports_cognition

# Standardize the column values by stripping whitespace and converting to lowercase
df_filtered["QID91"] = df_filtered["QID91"].astype(str).str.strip().str.lower() == "true"
df_filtered["Q89"] = df_filtered["Q89"].astype(str).str.strip().str.lower() == "true"
df_filtered["Q90"] = df_filtered["Q90"].astype(str).str.strip().str.lower() == "75 degrees or higher"

# Define the conditions for each attention check
condition_QID91 = df_filtered["QID91"] == True
condition_Q89 = df_filtered["Q89"] == True
condition_Q90 = df_filtered["Q90"] == True

# Count how many conditions each row meets
df_filtered["attention_pass_count"] = (
    condition_QID91.astype(int) + 
    condition_Q89.astype(int) + 
    condition_Q90.astype(int)
)

print(df_filtered[["QID91", "Q89", "Q90", "attention_pass_count"]].dtypes)

# Keep only rows where at least 2 out of 3 checks pass
df_filtered = df_filtered[df_filtered["attention_pass_count"] >= 2]

# Drop the attention check columns
df_filtered = df_filtered.drop(columns=["QID91", "Q89", "Q90"])

# Replace entries that don't match []
df_filtered["Q77"] = df_filtered["Q77"].str.strip().replace('[]', '')

# Replace entries in 'Q77' that do not start with '[' with ' '
df_filtered['Q77'] = df_filtered['Q77'].where(df_filtered['Q77'].str.startswith('['), ' ')

# Display the new shape of the dataset
print("New shape:", df_filtered.shape)

# Save the cleaned dataset
df_filtered.to_csv("./Data/Survey_data.csv", index=False)