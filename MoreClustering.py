import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create an output file to save console output
output_file = open("./Data/Clustering_output.txt", "w")


def print_and_log(message):
    """Print to console and write to output file"""
    print(message)
    output_file.write(str(message) + "\n")


# Load dataset
file_path = "./Data/Labeled_survey_data.csv"
df = pd.read_csv(file_path)


# Function to extract sport names from JSON-like structure
def extract_sport_names(entry):
    try:
        if isinstance(entry, str) and entry.startswith("["):
            parsed = ast.literal_eval(entry)  # Convert string to list of dicts
            if isinstance(parsed, list) and len(parsed) > 0 and "Sport" in parsed[0]:
                return parsed[0]["Sport"]
    except:
        return None
    return None


# Apply extraction function
df["Sport_Name"] = df["Sports_Info"].apply(extract_sport_names)

# Display unique sports to confirm extraction worked
print_and_log(f"Sport names distribution:\n{df['Sport_Name'].value_counts()}")

# Define the selected features
relevant_columns = [
    "Head_Injury_Status",
    "Concussion_Status",
    "Lose_Consciousness",
    "Bed_Time",
    "Min_To_Sleep",
    "Wake_Up",
    "Sleep_Per_Night",
    "Cant_Sleep",
    "Wake_In_Night",
    "Wake_To_Bathroom",
    "Breathe_Discomfort",
    "Snore_Cough",
    "Too_Cold",
    "Too_Hot",
    "Bad_Dreams",
    "Pain",
    "Sleep_Quality",
    "Sleep_Meds",
    "Staying_Awake_Issues",
    "Motivation_Issues",
    "Noise_Concentration_Issues",
    "Concentration_Issues",
    "Surrounding_Concentration_Issues",
    "Good_Music_Concentration",
    "Concentration_Aware",
    "Reading_Concentration_Issues",
    "Trouble_Blocking_Thoughts",
    "Excitement_Concentration_Issues",
    "Ignore_Hunger_Concentrating",
    "Good_Task_Switching",
    "Long_Time_Focus",
    "Poor_Listening_Writing",
    "Quick_Interest",
    "Easy_Read_Write_On_Phone",
    "Trouble_Multiple_Conversations",
    "Trouble_Quick_Creativity",
    "Good_Interruption_Recovery",
    "Good_Thought_Recovery",
    "Good_Task_Alteration",
    "Poor_Perspective_Thinking",
    "Sports_Concussion_Status",
]

# Keep only relevant columns
df_selected = df[relevant_columns].copy()

# Convert binary categorical data (Yes/No) to 0/1
binary_mappings = {"Yes": 1, "No": 0}
for col in [
    "Head_Injury_Status",
    "Concussion_Status",
    "Lose_Consciousness",
    "Sports_Concussion_Status",
]:
    df_selected[col] = df_selected[col].map(binary_mappings)


# Convert time-based features into numerical values
def convert_time_to_numeric(time_str):
    try:
        if isinstance(time_str, str) and ":" in time_str:
            parts = time_str.split(":")
            return int(parts[0]) + int(parts[1]) / 60  # Convert to decimal hours
        elif isinstance(time_str, str) and "Minute" in time_str:
            return float(time_str.split(" ")[0]) / 60  # Convert minutes to hours
        elif isinstance(time_str, str) and "Hour" in time_str:
            return float(time_str.split(" ")[0])  # Extract numeric part
        elif isinstance(time_str, (int, float)):
            return float(time_str)
    except:
        return np.nan
    return np.nan


# Apply conversion to time-based columns
time_columns = ["Bed_Time", "Wake_Up", "Min_To_Sleep", "Sleep_Per_Night"]
for col in time_columns:
    df_selected[col] = df_selected[col].apply(convert_time_to_numeric)

df_selected[time_columns] = df_selected[time_columns].apply(
    pd.to_numeric, errors="coerce"
)

# Mapping ordinal categorical responses
frequency_mappings = {
    "Not during the past month": 0,
    "Less than once a week": 1,
    "Once or twice a week": 2,
    "Three or more times a week": 3,
}
ordinal_mappings = {
    "Poor": 1,
    "Fair": 2,
    "Good": 3,
    "Very Good": 4,
    "Excellent": 5,
    "Almost never": 0,
    "Sometimes": 1,
    "Often": 2,
    "Always": 3,
}

for col in [
    "Cant_Sleep",
    "Wake_In_Night",
    "Wake_To_Bathroom",
    "Breathe_Discomfort",
    "Snore_Cough",
    "Too_Cold",
    "Too_Hot",
    "Bad_Dreams",
    "Pain",
    "Sleep_Meds",
    "Staying_Awake_Issues",
]:
    df_selected[col] = df_selected[col].map(frequency_mappings)

for col in [
    "Sleep_Quality",
    "Motivation_Issues",
    "Concentration_Issues",
    "Trouble_Blocking_Thoughts",
    "Good_Task_Switching",
    "Good_Thought_Recovery",
    "Good_Task_Alteration",
    "Poor_Perspective_Thinking",
    "Noise_Concentration_Issues",
    "Surrounding_Concentration_Issues",
    "Reading_Concentration_Issues",
    "Excitement_Concentration_Issues",
    "Ignore_Hunger_Concentrating",
    "Long_Time_Focus",
    "Poor_Listening_Writing",
    "Quick_Interest",
    "Easy_Read_Write_On_Phone",
    "Trouble_Multiple_Conversations",
    "Trouble_Quick_Creativity",
    "Good_Interruption_Recovery",
]:
    df_selected[col] = df_selected[col].map(ordinal_mappings)

# Ensure all columns are numeric before scaling
df_selected = df_selected.apply(pd.to_numeric, errors="coerce")

# Fill missing values
for col in df_selected.columns:
    if col in time_columns:
        df_selected[col] = df_selected[col].fillna(df_selected[col].median())
    else:
        if not df_selected[col].mode().empty:
            df_selected[col] = df_selected[col].fillna(df_selected[col].mode().iloc[0])
        else:
            df_selected[col] = df_selected[col].fillna(
                0
            )  # Default to 0 if no mode exists

# Normalize and apply PCA
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=df_selected.columns)
pca = PCA(n_components=10)
df_pca = pd.DataFrame(
    pca.fit_transform(df_scaled), columns=[f"PC{i+1}" for i in range(10)]
)

# Print the columns after PCA reduction
print_and_log("\nColumns after PCA reduction:")
print_and_log(df_pca.columns.tolist())

# Print explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
print_and_log(f"\nExplained variance by each principal component:")
for i, var in enumerate(explained_variance):
    print_and_log(f"PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")

# Apply clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
predictions = kmeans.fit_predict(df_pca)

print("Sihoulette Score " + str(silhouette_score(df_pca, predictions)))
df_pca["Cluster"] = predictions

# Add back Sport_Name and create final dataframe
df_clustered = df.loc[df_selected.index, ["Sport_Name"]].copy()
df_clustered["Cluster"] = df_pca["Cluster"]

# Merge PCA components into df_clustered
df_clustered = pd.concat([df_clustered, df_pca.iloc[:, :10]], axis=1)

# Show results
print_and_log("\nCluster distribution:")
print_and_log(df_clustered["Cluster"].value_counts())
print_and_log("\nSport distribution across clusters:")
sport_clusters = df_clustered.groupby("Cluster")["Sport_Name"].value_counts()
print_and_log(sport_clusters)

# Visualization 1: PCA scatter plot (PC1 vs PC2)
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="viridis",
    s=100,
    data=df_clustered,
    alpha=0.7,
)
plt.title("Cluster visualization using first two principal components", fontsize=15)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.legend(title="Cluster", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("Plots/pca_cluster_scatter.png")
print("\nSaved PCA scatter plot")

# Visualization 2: 3D PCA plot (PC1, PC2, PC3)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

# Create a color map with distinct colors for each cluster
colors = plt.cm.viridis(np.linspace(0, 1, 3))  # Sample 3 colors from viridis
clusters = sorted(df_clustered["Cluster"].unique())

# Plot each cluster with its own color and label
for cluster, color in zip(clusters, colors):
    cluster_data = df_clustered[df_clustered["Cluster"] == cluster]
    ax.scatter(
        cluster_data["PC1"],
        cluster_data["PC2"],
        cluster_data["PC3"],
        color=[color],
        s=60,
        alpha=0.8,
        label=f"Cluster {cluster}",
    )

# Configure the legend
ax.legend(
    title="Cluster",
    title_fontsize=14,  # Increases title font size
    fontsize=12,  # Increases label font size
    markerscale=1.75,  # Makes legend markers bigger
    frameon=True,  # Adds a frame around the legend
    framealpha=0.9,  # Makes the frame more opaque
    loc="upper right",  # Positions the legend
)
ax.set_title("3D Cluster Visualization (First 3 Principal Components)", fontsize=15)
ax.set_xlabel("PC1", fontsize=12)
ax.set_ylabel("PC2", fontsize=12)
ax.set_zlabel("PC3", fontsize=12)
plt.tight_layout()
plt.savefig("Plots/pca_3d_cluster.png")
print("\nSaved PCA 3D cluster")

# Visualization 3: Sport distribution by cluster
plt.figure(figsize=(14, 8))
df_sport_counts = (
    df_clustered.groupby(["Cluster", "Sport_Name"]).size().reset_index(name="Count")
)
df_sport_counts = df_sport_counts.dropna(
    subset=["Sport_Name"]
)  # Drop rows with NaN sport
pivot = df_sport_counts.pivot_table(
    values="Count", index="Sport_Name", columns="Cluster", fill_value=0
)

# Remove sports with very low counts for readability
pivot = pivot[
    pivot.sum(axis=1) >= 3
]  # Only include sports with at least 3 participants

ax = pivot.plot(kind="barh", stacked=True, figsize=(14, 8), colormap="viridis")
plt.title("Distribution of Sports by Cluster", fontsize=15)
plt.xlabel("Count", fontsize=12)
plt.ylabel("Sport", fontsize=12)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("Plots/sport_distribution_by_cluster.png")
print("\nSaved sport distribution visualization")

# Visualization 4: Cluster centers heatmap
plt.figure(figsize=(15, 8))
centers = pd.DataFrame(kmeans.cluster_centers_, columns=[f"PC{i+1}" for i in range(10)])
centers.index.name = "Cluster"
centers = centers.reset_index()
centers_melted = pd.melt(
    centers, id_vars=["Cluster"], var_name="Component", value_name="Value"
)

plt.figure(figsize=(12, 6))
heatmap = sns.heatmap(
    kmeans.cluster_centers_,
    annot=True,
    cmap="coolwarm",
    xticklabels=[f"PC{i+1}" for i in range(10)],
    yticklabels=[f"Cluster {i}" for i in range(3)],
)
plt.title("K-means Cluster Centers Across Principal Components", fontsize=15)
plt.xlabel("Principal Component", fontsize=12)
plt.ylabel("Cluster", fontsize=12)
plt.tight_layout()
plt.savefig("Plots/cluster_centers_heatmap.png")
print("\nSaved cluster centers heatmap")

# Visualization 5: Elbow method to verify optimal number of clusters
plt.figure(figsize=(10, 6))
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_test.fit(df_pca.iloc[:, :10])
    wcss.append(kmeans_test.inertia_)

plt.plot(k_range, wcss, marker="o", linestyle="-")
plt.title("Elbow Method For Optimal k", fontsize=15)
plt.xlabel("Number of Clusters", fontsize=12)
plt.ylabel("WCSS (Within-Cluster Sum of Squares)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("Plots/elbow_method.png")
print("\nSaved elbow method visualization")

# Save the clustered data to CSV
df_clustered.to_csv("Data/Clustered_data_results.csv", index=False)
print("\nSaved clustered data")

# Close the output file
print("\n All Plots Created")
output_file.close()
