from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from datetime import datetime
import re


df = pd.read_csv("./Data/Labeled_survey_data.csv")


model = KMeans(n_clusters=2)


def time_to_decimal(time_str):
    # Handle NaN or empty values
    if pd.isna(time_str):
        return np.nan

    # Convert to datetime
    time_obj = datetime.strptime(str(time_str), "%H:%M:%S")

    # Convert to decimal hours
    return time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600


def parse_minutes(duration_str):
    # Handle NaN values
    if pd.isna(duration_str):
        return np.nan

    # Convert to string to handle any type
    duration_str = str(duration_str).strip()

    # Extract numeric value using regex
    match = re.search(r"(\d+)\s*Minutes?", duration_str, re.IGNORECASE)

    if match:
        return int(match.group(1))
    else:
        # Return NaN if no match found
        return np.nan


def parse_hours(duration_str):
    # Handle NaN values
    if pd.isna(duration_str):
        return np.nan

    # Convert to string to handle any type
    duration_str = str(duration_str).strip()

    # Extract numeric value using regex
    # This will match numbers with optional decimal point (like 7.0 or 7.5)
    match = re.search(r"(\d+(?:\.\d+)?)\s*Hours?", duration_str, re.IGNORECASE)

    if match:
        return float(match.group(1))
    else:
        # Return NaN if no match found
        return np.nan


def frequency_to_number(frequency):
    # Handle NaN values
    if pd.isna(frequency):
        return np.nan

    # Convert to lowercase for consistent matching
    frequency = str(frequency).lower().strip()

    # Mapping of frequency descriptions to numerical values
    frequency_map = {
        "not during the past month": 0,
        "not during the past moth": 0,
        "less than once a week": 1,
        "once or twice a week": 2,
        "three or more times a week": 3,
    }

    # Return the corresponding numerical value
    return frequency_map.get(frequency, np.nan)


def quality_to_number(quality):
    # Handle NaN values
    if pd.isna(quality):
        return np.nan

    # Convert to lowercase for consistent matching
    quality = str(quality).lower().strip()

    # Mapping of frequency descriptions to numerical values
    quality_map = {
        "very bad": 0,
        "fairly bad": 1,
        "fairly good": 2,
        "very good": 3,
    }

    # Return the corresponding numerical value
    return quality_map.get(quality, np.nan)


class ClusterModel:
    model = KMeans(n_clusters=2)

    def fit(data)


"""
    Relevant Columns:
    Bed_Time
    Min_To_Sleep
    Wake_Up
    Sleep_Per_Night
    Cant_Sleep
    Wake_In_Night
    Wake_To_Bathroom
    Bad_Dreams
    Sleep_Quality
    Sleep_Meds
    Staying_Awake_Issues
    Loud_Snore
"""
# print(df.columns)

df["Bed_Time"] = df["Bed_Time"].apply(time_to_decimal)
df["Wake_Up"] = df["Wake_Up"].apply(time_to_decimal)


# Apply the parsing function
df["Min_To_Sleep"] = df["Min_To_Sleep"].apply(parse_minutes)
df["Sleep_Per_Night"] = df["Sleep_Per_Night"].apply(parse_hours)

# Print results to verify
df["Cant_Sleep"] = df["Cant_Sleep"].apply(frequency_to_number)
df["Wake_In_Night"] = df["Wake_In_Night"].apply(frequency_to_number)
df["Wake_To_Bathroom"] = df["Wake_To_Bathroom"].apply(frequency_to_number)
df["Bad_Dreams"] = df["Bad_Dreams"].apply(frequency_to_number)
df["Sleep_Quality"] = df["Sleep_Quality"].apply(quality_to_number)

df["Sleep_Meds"] = df["Sleep_Meds"].apply(frequency_to_number)
df["Staying_Awake_Issues"] = df["Staying_Awake_Issues"].apply(frequency_to_number)
df["Loud_Snore"] = df["Loud_Snore"].apply(frequency_to_number)
