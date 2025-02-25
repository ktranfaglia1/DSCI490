import pandas as pd
import re
from datetime import datetime

# from ParseSports import parse_sports  # to get sports JSONs dont use
# from ParseConcussions import parse_concussions  # to get sports JSONs dont use

# Read in dataset
df_sports_cognition = pd.read_csv("Data/Sport_Cognition.csv")

# Display original shape
# print("Original shape:", df_sports_cognition.shape)

# Display some data, statistics, and summary info
# print(df_sports_cognition.info())
# print(df_sports_cognition.describe())
# print("\n", df_sports_cognition.head())

# Check for missing values
# print(df_sports_cognition.isnull().sum())

# Drop columns where more than the threshold of values are missing
threshold = 0.9
df_filtered = df_sports_cognition.dropna(
    thresh=len(df_sports_cognition) * (1 - threshold), axis=1
)

# Remove rows where 'Finished' is False
df_filtered = df_filtered[df_filtered["Finished"] == "True"]

# Keep only rows where 'Q65' is "I am between the ages of 18-25."
df_filtered = df_filtered[df_filtered["Q65"] == "I am between the ages of 18-25."]


# Keep columns that start with "Q" or are "Duration (in seconds)"
df_filtered = df_filtered[
    [col for col in df_filtered.columns if col.startswith("Q")]
    + ["Duration (in seconds)"]
]


# Drop the "Q_RecaptchaScore" column
df_filtered = df_filtered.drop(columns=["Q_RecaptchaScore"], errors="ignore")

df_filtered = df_sports_cognition


def is_this(comparisonString, x):
    if type(x) is not str:
        return x
    return x.strip() == comparisonString


# print(df_filtered["Q89"])
df_filtered["QID91"] = df_filtered["QID91"].apply(lambda x: is_this("Spaceship", x))

df_filtered["Q89"] = df_filtered["Q89"].apply(lambda x: is_this("Kindergarten", x))

# Standardize the column values by stripping whitespace and converting to lowercase
df_filtered["QID91"] = (
    df_filtered["QID91"].astype(str).str.strip().str.lower() == "true"
)
df_filtered["Q89"] = df_filtered["Q89"].astype(str).str.strip().str.lower() == "true"
df_filtered["Q90"] = (
    df_filtered["Q90"].astype(str).str.strip().str.lower() == "75 degrees or higher"
)


# Define the conditions for each attention check
condition_QID91 = df_filtered["QID91"] == True
condition_Q89 = df_filtered["Q89"] == True
condition_Q90 = df_filtered["Q90"] == True


# Count how many conditions each row meets
df_filtered["attention_pass_count"] = (
    condition_QID91.astype(int) + condition_Q89.astype(int) + condition_Q90.astype(int)
)


# print(df_filtered[["QID91", "Q89", "Q90", "attention_pass_count"]].dtypes)

# Keep only rows where at least 2 out of 3 checks pass
df_filtered = df_filtered[df_filtered["attention_pass_count"] >= 2]

# Drop the attention check columns
df_filtered = df_filtered.drop(columns=["QID91", "Q89", "Q90"])

# Display the new shape of the dataset


def remove_parenthese(dataSeries: pd.Series) -> pd.Series:

    def cut_parenthese(x):
        if type(x) != str:
            return x
        return x.split("(")[0].strip()

    return dataSeries.apply(cut_parenthese)


def is_this(comparisonString, x):
    if type(x) is not str:
        return x
    return x.strip() == comparisonString


def get_first_number(text):

    if type(text) != str:
        return text
    # Regex pattern to match the first float or integer
    pattern = r"\d+(\.\d+)?"

    match = re.search(pattern, text)
    if match:
        return float(
            match.group(0)
        )  # Convert matched string to float (if it's a float or int)
    return None  # Return None if no match is found


def Q3_times(time):
    if type(time) != str:
        return time
    number = get_first_number(time)

    if type(number) != float:
        return number
    if number < 2:
        return number * 60  # returns time in minutes assuming hours
    return number


df_filtered["Q71"] = pd.to_datetime(df_filtered["Q71"], format="mixed", errors="coerce")


df_filtered["Q1"] = remove_parenthese(df_filtered["Q1"])
df_filtered["Q78"] = remove_parenthese(df_filtered["Q72"])
df_filtered["Q78"] = remove_parenthese(df_filtered["Q78"])
df_filtered["Q84"] = remove_parenthese(df_filtered["Q84"])


# standardize time format to 'hh:mm AM/PM'
def standardize_time(value):
    if pd.isna(value) or not isinstance(value, str) or value.strip() == "":
        return None

    value = value.strip().lower().replace(" ", "")

    # Handle missing AM/PM
    if value.isnumeric():
        hour = int(value)
        value += "am" if 1 <= hour <= 6 else "pm"

    if "am" in value or "pm" in value:
        if ":" not in value:
            value = value[:-2] + ":00" + value[-2:]

    try:
        return datetime.strptime(value, "%I:%M%p").strftime("%I:%M %p")
    except ValueError:
        return None


# Apply to bedtime (Q2) and wake-up time (Q4)
df_filtered["Q2"] = df_filtered["Q2"].apply(standardize_time)
df_filtered["Q4"] = df_filtered["Q4"].apply(standardize_time)


# convert sleep latency (Q3) to minutes
def standardize_minutes(value):
    if pd.isna(value) or not isinstance(value, str) or value.strip() == "":
        return None
    value = value.lower().strip().replace(" ", "")
    match = re.findall(r"(\d+)", value)
    if match:
        numbers = [int(n) for n in match]
        if "hr" in value or "hour" in value:
            numbers = [n * 60 for n in numbers]
        return sum(numbers) // len(
            numbers
        )  # Average if range, else return single value
    return None


# total sleep duration (Q5) to hours
def standardize_hours(value):
    if pd.isna(value) or not isinstance(value, str) or value.strip() == "":
        return None
    value = value.lower().strip().replace(" ", "")
    match = re.findall(r"(\d+\.?\d*)", value)
    if match:
        numbers = [float(n) for n in match]
        return sum(numbers) / len(numbers)
    return None


# Apply to Q3 and Q5
df_filtered["Q3"] = df_filtered["Q3"].apply(standardize_minutes)
df_filtered["Q5"] = df_filtered["Q5"].apply(standardize_hours)

# Append units
df_filtered["Q3"] = df_filtered["Q3"].apply(
    lambda x: f"{int(x)} Minutes" if pd.notna(x) else None
)
df_filtered["Q5"] = df_filtered["Q5"].apply(
    lambda x: f"{x:.1f} Hours" if pd.notna(x) else None
)

"""
df_filtered["Q77"] = df_filtered["Q77"].apply(
    parse_sports
)  # gets sports JSONs Dont use
"""

# gets sports JSONs Dont use

# Replace entries that don't match []
df_filtered["Q77"] = df_filtered["Q77"].str.strip().replace("[]", "")

# Replace entries in 'Q77' that do not start with '[' with ' '
df_filtered["Q77"] = df_filtered["Q77"].where(
    df_filtered["Q77"].str.startswith("["), " "
)


print(df_filtered.shape)
# Save the cleaned dataset
# df_filtered.to_csv("./Data/FilteredData.csv", index=False)
