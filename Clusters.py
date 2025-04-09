"""
    remove based on p value of .01 and .05
    fix feature displat columns
    Use Dylans new P Values

"""


import sys

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier


import pandas as pd
import numpy as np
from datetime import datetime
import re
from scipy import stats
import json
import typing


original_df = pd.read_csv("./Data/Labeled_survey_data.csv")
df = original_df

df = df[df["Sports_Info"] != ' '].reset_index(drop=False)
#print('|' + (df["Sports_Info"][327]) + '|')

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
        "almost never": 0,
        "no problem at all": 0,
        "less than once a week": 1,
        "sometimes": 1,
        "only a very slight problem": 1,
        "once or twice a week": 2,
        "often": 2,
        "somewhat of a problem": 2,
        "three or more times a week": 3,
        "always": 3,
        "a very big problem": 3,
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


"""
This Segment of code is dedicated to clustering Sleep issues
"""

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

"""
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
"""

columns = [
    "Bed_Time",
    "Wake_Up",
    "Min_To_Sleep",
    "Sleep_Per_Night",
    #    "Cant_Sleep",
    #    "Wake_In_Night",
    #   "Wake_To_Bathroom",
    "Bad_Dreams",
    "Sleep_Quality",
    "Sleep_Meds",
    "Staying_Awake_Issues",
    #    "Loud_Snore",
]

sleepData = df[columns]
# print(sleepData)


#print(sleepData)

numerical_sleep = sleepData.filter(["Bed_Time", "Wake_Up", "Min_To_Sleep", "Sleep_Per_Night"])
categorical_sleep = sleepData.filter(["Bad_Dreams", "Sleep_Quality", "Sleep_Meds", "Staying_Awake_Issues"])


num_imputer = SimpleImputer(strategy="mean")
numerical_sleep = num_imputer.fit_transform(numerical_sleep)

cat_encoder = OneHotEncoder(sparse_output=False)
categorical_sleep = cat_encoder.fit_transform(categorical_sleep)


categorical_sleep = pd.DataFrame(categorical_sleep.transpose(), cat_encoder.get_feature_names_out())
categorical_sleep = categorical_sleep.T
#print(encoder.get_feature_names_out())
#print(categorical_sleep)
#print(type(categorical_sleep))
keep_categories = ["Bad_Dreams_3.0", "Sleep_Meds_0.0", "Staying_Awake_Issues_0.0", "Staying_Awake_Issues_2.0"]
categorical_sleep = categorical_sleep.filter(keep_categories)
#print(categorical_sleep)


imputer = SimpleImputer(strategy="most_frequent")
categorical_sleep = imputer.fit_transform(categorical_sleep)

'''
print("Sleep Data")
print(sleepData)
print(encoder.get_feature_names_out())
'''


categorical_sleep = pd.DataFrame(categorical_sleep.transpose(), keep_categories)
categorical_sleep = categorical_sleep.T

numerical_sleep = pd.DataFrame(numerical_sleep.transpose(), num_imputer.get_feature_names_out())
numerical_sleep = numerical_sleep.T

'''
print(type(categorical_sleep))
print(type(numerical_sleep))
'''

sleepData = pd.merge(numerical_sleep, categorical_sleep, left_index=True, right_index=True)
columns = list(sleepData.columns)
#print(sleepData)
scaler = StandardScaler()
sleepData = scaler.fit_transform(sleepData)

# print(sleepData)

model = KMeans(
    n_clusters=2,
    init="k-means++",
    random_state=0,
    n_init=10,
).fit(sleepData)

forest = RandomForestClassifier()

forest.fit(sleepData, model.predict(sleepData))
important_features = forest.feature_importances_.argsort()[::-1]


print("Important Features Sleep: ")
for index in important_features:
    print(
        str(columns[index])
        + " Importance "
        + str(forest.feature_importances_[index])
    )

print()
print("Sleep Sihoulette: " + str(silhouette_score(sleepData, model.predict(sleepData))))

df["Sleep_Cluster"] = model.predict(sleepData)

# print(df)

"""
This Segment of code is dedicated to clustering Concentration issues
"""

columns = [
    "Motivation_Issues",
    #"Noise_Concentration_Issues",
    "Concentration_Issues",
    #"Good_Music_Concentration",
    #"Concentration_Aware", #.05/.01
    #"Reading_Concentration_Issues",
    #"Trouble_Blocking_Thoughts",
    #"Excitement_Concentration_Issues",
    #"Ignore_Hunger_Concentrating",
    "Good_Task_Switching",
    #"Long_Time_Focus",
    "Poor_Listening_Writing",
    #"Quick_Interest",
    #"Easy_Read_Write_On_Phone",#.05/.01
    #"Trouble_Multiple_Conversations",
    #"Trouble_Quick_Creativity",
    "Good_Interruption_Recovery",
    #"Good_Thought_Recovery",
    "Good_Task_Alteration",
    #"Poor_Perspective_Thinking",
]



attentionData = df[columns]
# print(sleepData)



for column in columns:
    attentionData.loc[:, column] = attentionData[column].apply(frequency_to_number)

#print(attentionData)
'''
encoder = OneHotEncoder(sparse_output=False)
attentionData = encoder.fit_transform(attentionData)
print(encoder.get_feature_names_out())
'''

print("\n\n\n\n")

column_stuff = attentionData.columns
imputer = SimpleImputer(strategy="most_frequent")
attentionData = imputer.fit_transform(attentionData)

att_encoder = OneHotEncoder(sparse_output=False)
categorical_attention = att_encoder.fit_transform(attentionData)


categorical_attention = pd.DataFrame(categorical_attention.transpose(), att_encoder.get_feature_names_out())
categorical_attention = categorical_attention.T

rename_dict = {}
for col in categorical_attention.columns:
    backend = col[2:]
    index = int(col[1])
    rename_dict[col] =  column_stuff[index] + backend

categorical_attention = categorical_attention.rename(columns=rename_dict)
#print(encoder.get_feature_names_out())
#print(categorical_sleep)
#print(type(categorical_sleep))

#keep_categories = ["Good_Interruption_Recovery_0.0", "Good_Task_Switching_3.0"] #.01
keep_categories = ["Good_Interruption_Recovery_0.0", "Good_Task_Switching_2.0", "Concentration_Issues_3.0", "Good_Task_Alteration_3.0", "Poor_Listening_Writing_3.0"] #.05
attentionData = categorical_attention.filter(keep_categories)
feature_columns = attentionData.columns
scaler = StandardScaler()
attentionData = scaler.fit_transform(attentionData)

# attentionData = attentionData[(np.abs(stats.zscore(attentionData["feature"])) < 3)]
# print(sleepData)


model = KMeans(n_clusters=2, init="k-means++", random_state=0).fit(attentionData)

forest = RandomForestClassifier()




forest.fit(attentionData, model.predict(attentionData))
important_features = forest.feature_importances_.argsort()[::-1]



print("Important Features Attention: ")
for index in important_features:
    print(
        str(feature_columns[index])
        + " Importance "
        + str(forest.feature_importances_[index])
    )


print()
print(
    "Attention Sihoulette: "
    + str(silhouette_score(attentionData, model.predict(attentionData)))
)

df["Attention_Cluster"] = model.predict(attentionData)

#print(df["Sports_Concussion_Info"])
new_column = []
for concussion_json in df["Sports_Concussion_Info"]:
    
    if type(concussion_json) is not float:
        concussion_json = json.loads(concussion_json)
        num_concussion = 0
        #print(concussion_json)
        for sport in concussion_json:
            num_concussion += int(sport["Concussions"])
        
        new_column.append(num_concussion)
    else:
        new_column.append(0)

df["Num_Concussions"] = new_column

print(len(df))
print()
print(df.groupby("Sleep_Cluster")["Num_Concussions"].describe())
print()
print(df.groupby("Attention_Cluster")["Num_Concussions"].describe())



player_sport = []
player_cluster = []


for index, player in enumerate(df["Sports_Info"]):
    if player != ' ':
        sport_json = json.loads(player)
        for sport in sport_json:
            player_sport.append(sport["Sport"])
            player_cluster.append(df["Sleep_Cluster"][index])
    


indexes = [i for i in range(len(player_cluster))]

sport_array = pd.Series(player_sport[i] for i in range(len(player_sport)))
cluster_array = pd.Series(player_cluster[i] for i in range(len(player_cluster)))

sport_cluster_df = pd.DataFrame({
    "sport": sport_array,
    "cluster": cluster_array
})

sport_cluster_df['sport'] = sport_cluster_df['sport'].replace(
    {'competitive dance': 'dance', 'dancer': 'dance', 'dancing' : 'dance', 'gymnast' : 'gymnastics'}
)


print("\n\n\n")
sport_cluster_df = sport_cluster_df.groupby("sport")["cluster"].apply(list)


print("Sleep Clusters")
print("Cluster 0")
for cluster, sport_name in zip(sport_cluster_df, sport_cluster_df.index):
    if cluster.count(0) > cluster.count(1):
        print(f"{sport_name}: {round(cluster.count(0) / len(cluster) * 100, 2)}%, {cluster.count(0)}:{cluster.count(1)}")

print("\n")
print("Cluster 1")
for cluster, sport_name in zip(sport_cluster_df, sport_cluster_df.index):
    if cluster.count(1) >= cluster.count(0):
        print(f"{sport_name}: {round(cluster.count(1) / len(cluster) * 100, 2)}%, {cluster.count(0)}:{cluster.count(1)}")



player_sport = []
player_cluster = []

for index, player in enumerate(df["Sports_Info"]):
    if player != ' ':
        sport_json = json.loads(player)
        for sport in sport_json:
            player_sport.append(sport["Sport"])
            player_cluster.append(df["Attention_Cluster"][index])


indexes = [i for i in range(len(player_cluster))]

sport_array = pd.Series(player_sport[i] for i in range(len(player_sport)))
cluster_array = pd.Series(player_cluster[i] for i in range(len(player_cluster)))

sport_cluster_df = pd.DataFrame({
    "sport": sport_array,
    "cluster": cluster_array
})

sport_cluster_df['sport'] = sport_cluster_df['sport'].replace(
    {'competitive dance': 'dance', 'dancer': 'dance', 'dancing' : 'dance', 'gymnast' : 'gymnastics'}
)


print("\n\n\n")
sport_cluster_df = sport_cluster_df.groupby("sport")["cluster"].apply(list)


print("Attention Clusters")
print("Cluster 0")
for cluster, sport_name in zip(sport_cluster_df, sport_cluster_df.index):
    if cluster.count(0) > cluster.count(1):
        print(f"{sport_name}: {round(cluster.count(0) / len(cluster) * 100, 2)}%, {cluster.count(0)}:{cluster.count(1)}")

print("\n")
print("Cluster 1")
for cluster, sport_name in zip(sport_cluster_df, sport_cluster_df.index):
    if cluster.count(1) >= cluster.count(0):
        print(f"{sport_name}: {round(cluster.count(1) / len(cluster) * 100, 2)}%, {cluster.count(0)}:{cluster.count(1)}")


#original_df.to_csv("./Data/Labeled_survey_data.csv", index=False)

print()
