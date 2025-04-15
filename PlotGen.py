import pandas as pd
import matplotlib.pyplot as plt
import json

'''
    Sleep_Cluster_05: 
        Collision -> 0
        Contact -> 1

    Sleep_Cluster_01: 
        Collision -> 0
        Contact -> 1

    Attention_Cluster_05: 
        Collision -> 0
        Contact -> 1

    Attention_Cluster_01: 
        Contact -> 0
        Collison -> 1
'''

df = pd.read_csv("./Data/Labeled_survey_data.csv")



'''
Cluster Concussion Bar Charts
cluster_columns = ["Sleep_Cluster_05", "Sleep_Cluster_01", "Attention_Cluster_05", "Attention_Cluster_01", "combined_05", "combined_01"]


fig, axs = plt.subplots(2, 3)

for index, column in enumerate(cluster_columns):
    for classification in range(2):
        axs[index % 2][index // 2].bar(classification, df[(df[column] == classification)]["Num_Concussions"].mean())
    axs[index % 2][index // 2].set_title(column)
    if((index + 1) % 2 == 0):
        axs[index % 2][index // 2].set_xticks([0, 1])
    else:
        axs[index % 2][index // 2].set_xticks([])
    if (index // 2 == 0):
        axs[index % 2][index // 2].set_ylabel("Avg. # of Concussions")
        
    else:
        axs[index % 2][index // 2].set_yticklabels([])
        axs[index % 2][index // 2].set_ylabel("")
    
    axs[index % 2][index // 2].set_ylim(0, 0.5)  # Standardize y-axis
    plt.tight_layout()
plt.savefig("./Plots/Dustins/ClusterConcussions.png")
'''

"""
    Sports Clusters Pie Chart


def get_sport(sport_jstring: str) -> str:
    if sport_jstring != ' ':
        sport_json = json.loads(sport_jstring)
        if sport_json[0]["Sport"]:
            return sport_json[0]["Sport"]
        else:
            return None
    else:
        return None


columns = ["Sleep_Cluster_01", "Attention_Cluster_05"]
#print(df.columns)


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        count = int(round(pct * total / 100.0))
        return f'{count}'
    return my_autopct

for cIndex, column in enumerate(columns):
    
    df = df.filter(["Sports_Info"] + columns)
    df["Sport"] = df["Sports_Info"].apply(get_sport)

    for clusterIndex in range(2):
        df_temp = df[df[column] == clusterIndex]
        counts = df_temp["Sport"].value_counts()
        df_temp["Sport"] = df_temp["Sport"].apply(lambda x: x if x in counts and counts[x] >= 3 else "Other")
        plt.pie(df_temp["Sport"].value_counts(), labels=df_temp["Sport"].value_counts().index, autopct=make_autopct(counts.values))
        plt.title(f"Dataset: {column} Cluster {clusterIndex}")
        plt.savefig(f"./Plots/Dustins/Pie{column}_cluster_{clusterIndex}.png")
        plt.clf()

"""