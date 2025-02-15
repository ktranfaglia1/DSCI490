import typing
import pandas as pd
import numpy as np

#from parseSports import parse_sports  # to get sports JSONs dont use

initialDataset = pd.read_csv("./Data/cleaned_SP_data.csv")

filteredData = initialDataset


def remove_parenthese(dataSeries: pd.Series) -> pd.Series:

    def cut_parenthese(x):
        if type(x) != str:
            return np.nan
        return x.split("(")[0].strip()

    return dataSeries.apply(cut_parenthese)


def is_this(comparisonString, x):
    if type(x) is not str:
        return np.nan
    return x.strip() == comparisonString


filteredData["Q71"] = pd.to_datetime(
    filteredData["Q71"], format="mixed", errors="coerce"
)

filteredData["Q1"] = remove_parenthese(filteredData["Q1"])
filteredData["Q78"] = remove_parenthese(filteredData["Q72"])
filteredData["Q78"] = remove_parenthese(filteredData["Q78"])
filteredData["Q84"] = remove_parenthese(filteredData["Q84"])

filteredData["QID91"] = filteredData["QID91"].apply(lambda x: is_this("Spaceship", x))
filteredData["Q89"] = filteredData["Q89"].apply(lambda x: is_this("Kindergarten", x))

#filteredData["Q77"] = filteredData["Q77"].apply(
    parse_sports
)  # gets sports JSONs Dont use

filteredData.to_csv("./Data/cleaned_SP_data_test.csv")
