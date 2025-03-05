from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


df = pd.read_csv("./Data/Labeled_survey_data.csv")


model = KMeans(n_clusters=2)


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
print(df.columns)
print(df["Bed_Time"])
