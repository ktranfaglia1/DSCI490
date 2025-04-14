import pandas as pd
import matplotlib.pyplot as plt

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

