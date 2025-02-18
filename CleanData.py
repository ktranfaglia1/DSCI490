import typing
import pandas as pd
import numpy as np
from dateutil import parser
import re

# from parseSports import parse_sports  # to get sports JSONs dont use

initialDataset = pd.read_csv("./Data/cleaned_SP_data_test.csv")

filteredData = initialDataset


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


filteredData["Q71"] = pd.to_datetime(
    filteredData["Q71"], format="mixed", errors="coerce"
)

filteredData["Q1"] = remove_parenthese(filteredData["Q1"])
filteredData["Q78"] = remove_parenthese(filteredData["Q72"])
filteredData["Q78"] = remove_parenthese(filteredData["Q78"])
filteredData["Q84"] = remove_parenthese(filteredData["Q84"])

filteredData["QID91"] = filteredData["QID91"].apply(lambda x: is_this("Spaceship", x))
filteredData["Q89"] = filteredData["Q89"].apply(lambda x: is_this("Kindergarten", x))

# filteredData["Q77"] = filteredData["Q77"].apply(
#    parse_sports
# )  # gets sports JSONs Dont use

filteredData["Q3"] = filteredData["Q3"].apply(Q3_times)
filteredData["Q5"] = filteredData["Q5"].apply(get_first_number)


filteredData.to_csv("./Data/cleaned_SP_data_test.csv")
