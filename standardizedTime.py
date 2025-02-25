#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
from datetime import datetime

# Load dataset
df = pd.read_csv("./Data/FilteredData.csv")


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
df["Q2"] = df["Q2"].apply(standardize_time)
df["Q4"] = df["Q4"].apply(standardize_time)


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
df["Q3"] = df["Q3"].apply(standardize_minutes)
df["Q5"] = df["Q5"].apply(standardize_hours)

# Append units
df["Q3"] = df["Q3"].apply(lambda x: f"{int(x)} Minutes" if pd.notna(x) else None)
df["Q5"] = df["Q5"].apply(lambda x: f"{x:.1f} Hours" if pd.notna(x) else None)

# Save the final standardized dataset
df.to_csv("Standardized_SP_data.csv", index=False)


# In[ ]:
