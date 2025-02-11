# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read in dataset 
df_sports_cognition = pd.read_csv("Data/Sport_Cognition.csv")

# Display some data, statistics, and summary info
print(df_sports_cognition.info())
print(df_sports_cognition.describe())
print("\n", df_sports_cognition.head())

# Check for missing values
print(df_sports_cognition.isnull().sum())