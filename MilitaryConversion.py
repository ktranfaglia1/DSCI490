from datetime import datetime
import pandas as pd
import numpy as np


def convert_time(time_str):
    # Handle NaN or None values
    if pd.isna(time_str):
        return np.nan

    # Convert to string to handle potential float inputs
    time_str = str(time_str).strip()

    # Try different possible input formats
    time_formats = [
        "%I:%M %p",  # "10:00 PM"
        "%I:%M:%S %p",  # "10:00:00 PM"
        "%I:%M%p",  # "10:00PM"
        "%I:%M:%S%p",  # "10:00:00PM"
        "%I:%M %p",  # "1:00 AM"
    ]

    for fmt in time_formats:
        try:
            return datetime.strptime(time_str, fmt).strftime("%H:%M:%S")
        except ValueError:
            continue

    # If no format works, return NaN
    return np.nan


# Read the CSV
df = pd.read_csv("./Data/Labeled_survey_data.csv")

# Print original times
print("Original Times:")
print(df["Bed_Time"])

# Convert times
df["Bed_Time"] = df["Bed_Time"].apply(convert_time)
df["Wake_Up"] = df["Wake_Up"].apply(convert_time)

# Print converted times
print("\nConverted Times:")
print(df["Bed_Time"])

# Optional: Save the modified DataFrame
df.to_csv("./Data/Labeled_survey_data.csv")
# df.to_csv("./Data/Labeled_survey_data_converted.csv", index=False)
