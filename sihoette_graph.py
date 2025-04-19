from matplotlib import pyplot as plt

columns = ["Combined_05", "Combined_01", "Attention_01", "Attention_05", "Sleep_05", "Sleep_01"]
values = [.36, .77 ,.81, .53, .18 , .27]


plt.ylim(0,1)

plt.xlabel("Cluster Features")
plt.ylabel("Silhouette Score")

plt.xticks(rotation=45)

plt.bar(columns, values)
plt.tight_layout()
#plt.show()
plt.savefig("./Plots/Narrative/2_Cluster_Count/Sihoette_scores")