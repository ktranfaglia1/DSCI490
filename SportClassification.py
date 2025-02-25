from datetime import datetime
import pandas as pd

df = pd.read_csv("./Data/FilteredData2.csv")


def to_military_time(x):
    if isinstance(x, str):  # Check if the value is a string
        try:
            return datetime.strptime(x, "%I:%M %p").strftime("%H:%M")
        except ValueError:
            return x  # If the string can't be parsed, return it unchanged
    return x  # Return the value unchanged if it's not a string


# df["Q2"] = df["Q2"].apply(lambda x: to_military_time(x))
df["Q4"] = df["Q4"].apply(lambda x: to_military_time(x))

df.to_csv("./Data/FilteredData2.csv", index=False)
