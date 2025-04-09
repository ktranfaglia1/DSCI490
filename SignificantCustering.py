import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from collections import Counter
import warnings

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# Create an output file to save console output
output_file = open("./Data/Significant_Clustering_output.txt", "w")

def print_and_log(message):
    print(message)
    output_file.write(str(message) + "\n")

print_and_log("CLUSTERING WITH STATISTICALLY SIGNIFICANT FEATURES\n" + "="*50)

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
print_and_log(f"Sport names distribution:\n{df['Sport_Name'].value_counts().head(10)}")
print_and_log(f"Total unique sports: {df['Sport_Name'].nunique()}")

# Define only the statistically significant features (p < 0.05)
significant_columns = [
    # Sleep Features
    "Bad_Dreams",  # Response 3
    "Cant_Sleep",  # Response 3
    "Loud_Snore",  # Response 0
    "Sleep_Meds",  # Response 0
    "Sleep_Quality",  # Response 1
    "Staying_Awake_Issues",  # Response 0, 2
    "Wake_In_Night",  # Response 2
    "Wake_To_Bathroom",  # Response 0, 3
    
    # Attention Features
    "Concentration_Issues",  # Always
    "Good_Interruption_Recovery",  # Almost Never
    "Good_Task_Alteration",  # Almost Never
    "Good_Task_Switching",  # Often
    "Poor_Listening_Writing",  # Always
    
    # # Include important demographic/status variables
    # "Head_Injury_Status", 
    # "Concussion_Status", 
    # "Sports_Concussion_Status",
    # "Lose_Consciousness"
]

# Keep only significant columns
df_selected = df[significant_columns].copy()

# # Convert binary categorical data (Yes/No) to 0/1
# binary_mappings = {"Yes": 1, "No": 0}
# for col in ["Head_Injury_Status", "Concussion_Status", "Sports_Concussion_Status", "Lose_Consciousness"]:
#     df_selected[col] = df_selected[col].map(binary_mappings)

# Mapping ordinal categorical responses
ordinal_mappings = {
    "Poor": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5,
    "Almost never": 0, "Sometimes": 1, "Often": 2, "Always": 3
}

for col in ["Concentration_Issues", "Good_Interruption_Recovery", 
            "Good_Task_Alteration", "Good_Task_Switching", "Poor_Listening_Writing"]:
    df_selected[col] = df_selected[col].map(ordinal_mappings)

# Create binary features for the specific response categories that were significant
print_and_log("\nCreating binary features for significant response categories...")

# For sleep features with significant specific responses
df_selected['Bad_Dreams_Response3'] = (df_selected['Bad_Dreams'] == 3).astype(int)
df_selected['Cant_Sleep_Response3'] = (df_selected['Cant_Sleep'] == 3).astype(int)
df_selected['Loud_Snore_Response0'] = (df_selected['Loud_Snore'] == 0).astype(int)
df_selected['Sleep_Meds_Response0'] = (df_selected['Sleep_Meds'] == 0).astype(int)
df_selected['Sleep_Quality_Response1'] = (df_selected['Sleep_Quality'] == 1).astype(int)
df_selected['Staying_Awake_Response0'] = (df_selected['Staying_Awake_Issues'] == 0).astype(int)
df_selected['Staying_Awake_Response2'] = (df_selected['Staying_Awake_Issues'] == 2).astype(int)
df_selected['Wake_In_Night_Response2'] = (df_selected['Wake_In_Night'] == 2).astype(int)
df_selected['Wake_To_Bathroom_Response0'] = (df_selected['Wake_To_Bathroom'] == 0).astype(int)
df_selected['Wake_To_Bathroom_Response3'] = (df_selected['Wake_To_Bathroom'] == 3).astype(int)

# For attention features with significant specific responses
df_selected['Concentration_Always'] = (df_selected['Concentration_Issues'] == 3).astype(int)
df_selected['Interruption_Almost_Never'] = (df_selected['Good_Interruption_Recovery'] == 0).astype(int)
df_selected['Task_Alteration_Almost_Never'] = (df_selected['Good_Task_Alteration'] == 0).astype(int)
df_selected['Task_Switching_Often'] = (df_selected['Good_Task_Switching'] == 2).astype(int)
df_selected['Listening_Writing_Always'] = (df_selected['Poor_Listening_Writing'] == 3).astype(int)

# Display summary of the new binary features
print_and_log("\nBinary features for significant response categories created:")
binary_features = [col for col in df_selected.columns if col not in significant_columns]
for feature in binary_features:
    print_and_log(f"{feature}: Count = {df_selected[feature].sum()}, Percentage = {df_selected[feature].mean()*100:.2f}%")

# Ensure all columns are numeric before scaling
df_selected = df_selected.apply(pd.to_numeric, errors='coerce')

# Fill missing values with more sophisticated methods
for col in df_selected.columns:
    if df_selected[col].isna().sum() > 0:
        # For all variables, use mode (most frequent value)
        if not df_selected[col].mode().empty:
            df_selected[col] = df_selected[col].fillna(df_selected[col].mode().iloc[0])
        else:
            df_selected[col] = df_selected[col].fillna(0)  # Default to 0 if no mode exists

# Only use the binary features for clustering and drop the original features
# df_for_clustering = df_selected[binary_features + ["Head_Injury_Status", "Concussion_Status", "Sports_Concussion_Status"]].copy()
df_for_clustering = df_selected[binary_features].copy()

print_and_log(f"\nUsing {len(df_for_clustering.columns)} features for clustering:")
for col in df_for_clustering.columns:
    print_and_log(f"  - {col}")

# Use RobustScaler to scale the features
print_and_log("\nScaling data using RobustScaler to minimize outlier impact...")
robust_scaler = RobustScaler()
df_scaled = pd.DataFrame(robust_scaler.fit_transform(df_for_clustering), columns=df_for_clustering.columns)

# Determine optimal number of PCs based on explained variance
pca_full = PCA()
pca_full.fit(df_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find number of components that explain at least 80% of variance
n_components = np.argmax(cumulative_variance >= 0.80) + 1
print_and_log(f"\nOptimal number of PCs explaining ≥ 80% variance: {n_components}")

# Apply PCA with optimal components
pca = PCA(n_components=n_components)
df_pca = pd.DataFrame(
    pca.fit_transform(df_scaled), 
    columns=[f"PC{i+1}" for i in range(n_components)]
)

# Print explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
print_and_log(f"\nExplained variance by each principal component:")
for i, var in enumerate(explained_variance):
    print_and_log(f"PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")

# Visualize feature contributions to principal components
component_loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(n_components)],
    index=df_scaled.columns
)

# Plot heatmap of feature loadings for the first 5 PCs (or all if fewer than 5)
plt.figure(figsize=(16, 14))
num_pcs_to_show = min(5, n_components)
loadings_heatmap = sns.heatmap(
    component_loadings.iloc[:, :num_pcs_to_show],
    cmap="coolwarm", 
    annot=True, 
    fmt=".2f",
    cbar_kws={"label": "Feature Contribution"}
)
plt.title('Feature Contributions to Principal Components (Significant Features Only)', fontsize=16)
plt.tight_layout()
plt.savefig("Plots/pca_significant_feature_loadings.png")
print("\nSaved PCA feature loadings heatmap")

# Determine optimal number of clusters using multiple metrics
print_and_log("\nDetermining optimal number of clusters...")
k_range = range(2, 6)  # Test 2-5 clusters
silhouette_scores = []
ch_scores = []
db_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_pca)
    
    silhouette_scores.append(silhouette_score(df_pca, labels))
    ch_scores.append(calinski_harabasz_score(df_pca, labels))
    db_scores.append(davies_bouldin_score(df_pca, labels))

# Plot metrics for different cluster numbers
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
ax[0].plot(k_range, silhouette_scores, 'o-')
ax[0].set_xlabel('Number of clusters')
ax[0].set_ylabel('Silhouette Score')
ax[0].set_title('Silhouette Score (higher is better)')
ax[0].grid(True)

ax[1].plot(k_range, ch_scores, 'o-')
ax[1].set_xlabel('Number of clusters')
ax[1].set_ylabel('Calinski-Harabasz Score')
ax[1].set_title('Calinski-Harabasz Score (higher is better)')
ax[1].grid(True)

ax[2].plot(k_range, db_scores, 'o-')
ax[2].set_xlabel('Number of clusters')
ax[2].set_ylabel('Davies-Bouldin Score')
ax[2].set_title('Davies-Bouldin Score (lower is better)')
ax[2].grid(True)

plt.tight_layout()
plt.savefig("Plots/significant_cluster_metrics.png")
print("\nSaved cluster evaluation metrics")

# Find optimal k based on metrics
best_k_silhouette = k_range[np.argmax(silhouette_scores)]
best_k_ch = k_range[np.argmax(ch_scores)]
best_k_db = k_range[np.argmin(db_scores)]

print_and_log(f"\nOptimal number of clusters:")
print_and_log(f"Based on Silhouette Score: {best_k_silhouette}")
print_and_log(f"Based on Calinski-Harabasz Index: {best_k_ch}")
print_and_log(f"Based on Davies-Bouldin Index: {best_k_db}")

# Determine final k based on the majority vote or your suggestion
k = Counter([best_k_silhouette, best_k_ch, best_k_db]).most_common(1)[0][0]
print_and_log(f"\nSelected number of clusters based on majority vote: {k}")

# Perform clustering with optimal k and 2 clusters
print_and_log(f"\n{'='*20} ANALYSIS WITH {k} CLUSTERS {'='*20}")

# Apply K-Means with the optimal number of clusters
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_pca[f"Cluster_{k}"] = kmeans.fit_predict(df_pca)

# Add back Sport_Name and create final dataframe
df_clustered = df.loc[df_for_clustering.index, ["Sport_Name"]].copy()
df_clustered[f"Cluster_{k}"] = df_pca[f"Cluster_{k}"]

# Merge PCA components into df_clustered
df_clustered = pd.concat([df_clustered, df_pca.drop(columns=[f"Cluster_{k}"])], axis=1)

# Calculate cluster evaluation metrics
print_and_log(f"\nCluster evaluation metrics for k={k}:")
silhouette = silhouette_score(df_pca.drop(columns=[f"Cluster_{k}"]), df_pca[f"Cluster_{k}"])
ch_index = calinski_harabasz_score(df_pca.drop(columns=[f"Cluster_{k}"]), df_pca[f"Cluster_{k}"])
db_index = davies_bouldin_score(df_pca.drop(columns=[f"Cluster_{k}"]), df_pca[f"Cluster_{k}"])

print_and_log(f"Silhouette Score: {silhouette:.4f} (higher is better)")
print_and_log(f"Calinski-Harabasz Index: {ch_index:.4f} (higher is better)")
print_and_log(f"Davies-Bouldin Index: {db_index:.4f} (lower is better)")

# Show results
print_and_log(f"\nCluster distribution for k={k}:")
print_and_log(df_clustered[f"Cluster_{k}"].value_counts())
print_and_log(f"\nSport distribution across {k} clusters:")
sport_clusters = df_clustered.groupby([f"Cluster_{k}"])["Sport_Name"].value_counts().head(20)
print_and_log(sport_clusters)

# Add original features to the clusters for interpretation
cluster_features = pd.concat([df_for_clustering, df_clustered[f"Cluster_{k}"]], axis=1)

# Calculate and display cluster profiles
print_and_log(f"\nCluster profiles for k={k}:")
cluster_profiles = cluster_features.groupby(f"Cluster_{k}").mean()
print_and_log(cluster_profiles)

# Focus on the most distinctive features between clusters
feature_importance = {}
for feature in cluster_profiles.columns:
    feature_variance = cluster_profiles[feature].var()
    if not np.isnan(feature_variance):
        feature_importance[feature] = feature_variance

# Get top 10 most distinguishing features
top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
top_feature_names = [feature[0] for feature in top_features]

print_and_log("\nTop 10 most distinguishing features between clusters:")
for feature, importance in top_features:
    print_and_log(f"{feature}: {importance:.4f}")

# Display cluster means for top features
print_and_log("\nCluster means for top distinguishing features:")
print_and_log(cluster_profiles[top_feature_names])

# Visualization with t-SNE for better cluster separation
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
df_tsne = pd.DataFrame(
    tsne.fit_transform(df_pca.drop(columns=[f"Cluster_{k}"])),
    columns=['t-SNE-1', 't-SNE-2']
)
df_tsne[f"Cluster_{k}"] = df_pca[f"Cluster_{k}"]

# Visualization 1: t-SNE scatter plot
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x="t-SNE-1", 
    y="t-SNE-2", 
    hue=f"Cluster_{k}",
    palette="viridis",
    s=100,
    data=df_tsne,
    alpha=0.7
)
plt.title(f't-SNE Visualization of {k} Clusters (Significant Features)', fontsize=15)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.legend(title='Cluster', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"Plots/tsne_significant_cluster_{k}_scatter.png")
print(f"\nSaved t-SNE scatter plot for {k} clusters")

# Visualization 2: PCA scatter plot (PC1 vs PC2)
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x="PC1", 
    y="PC2", 
    hue=f"Cluster_{k}",
    palette="viridis",
    s=100,
    data=df_clustered,
    alpha=0.7
)
plt.title(f'PCA Visualization of {k} Clusters (Significant Features)', fontsize=15)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Cluster', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"Plots/pca_significant_cluster_{k}_scatter.png")
print(f"\nSaved PCA scatter plot for {k} clusters")

# Visualization 3: Spider plot of cluster profiles for the top features
plt.figure(figsize=(14, 10))

# Normalize the feature values for the spider plot
profile_norm = (cluster_profiles[top_feature_names] - cluster_profiles[top_feature_names].min()) / \
                (cluster_profiles[top_feature_names].max() - cluster_profiles[top_feature_names].min())

# Number of variables
categories = top_feature_names
N = len(categories)

# Create angle for each feature
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Initialize the spider plot
ax = plt.subplot(111, polar=True)

# Draw one axis per variable and add labels
plt.xticks(angles[:-1], categories, size=12)

# Draw the cluster profile lines
for i in range(k):
    values = profile_norm.iloc[i].values.tolist()
    values += values[:1]  # Close the loop
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Cluster {i}")
    ax.fill(angles, values, alpha=0.1)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title(f'Cluster Profiles Comparison for {k} Clusters (Significant Features)', size=15)
plt.tight_layout()
plt.savefig(f"Plots/cluster_{k}_significant_profiles_spider.png")
print(f"\nSaved cluster profiles spider plot for {k} clusters")

# Save the clustered data to CSV
df_clustered.to_csv(f"Data/Significant_Clustered_data_{k}_clusters.csv", index=False)
print(f"\nSaved clustered data for {k} clusters")

# Define collision sports (typically involve intentional body-to-body collisions)
collision_sports = [
    'Football', 'Rugby', 'Hockey', 'Ice Hockey', 'Street Hockey',
    'Wrestling', 'Lacrosse', 'Cheer'
]

# Define contact sports (including limited-contact sports)
contact_sports = [
    # Regular contact sports
    'Basketball', 'Soccer', 'Baseball', 'Volleyball', 'Tennis', 'Field Hockey',
    'Karate', 'TaeKwonDo', 'Gymnastics', 'Netball',
    'Bowling', 'Dance', 'Twirling',
    # Limited-contact sports
    'Track', 'Swim', 'Golf', 'Cross Country', 'Rowing', 'Crew', 'Marching Band',
    'Archery', 'Horseback Riding', 'Fencing'
]

# Calculate percentage of each sport type in each cluster
collision_counts = {0: 0, 1: 0}
contact_counts = {0: 0, 1: 0}
total_identified = {0: 0, 1: 0}
unidentified_counts = {0: 0, 1: 0}

# Loop through each sport in each cluster
for cluster in [0, 1]:
    sports_in_cluster = df_clustered[df_clustered["Cluster_2"] == cluster]["Sport_Name"].value_counts()
    
    for sport, count in sports_in_cluster.items():
        if pd.notna(sport):
            # Standardize sport name for comparison
            std_sport = sport.lower().strip()
            
            # Check sport type through case-insensitive comparison
            is_collision = any(collision.lower() == std_sport for collision in collision_sports)
            is_contact = any(contact.lower() == std_sport for contact in contact_sports)
            
            # If exact match failed, try partial match
            if not (is_collision or is_contact):
                is_collision = any(collision.lower() in std_sport for collision in collision_sports)
                is_contact = any(contact.lower() in std_sport for contact in contact_sports)
            
            if is_collision:
                collision_counts[cluster] += count
                total_identified[cluster] += count
            elif is_contact:
                contact_counts[cluster] += count
                total_identified[cluster] += count
            else:
                unidentified_counts[cluster] += count

# After calculating collision_counts, contact_counts, and total_identified for each cluster
# Add these calculations for overall percentages:

# Calculate total numbers across all clusters
total_collision = sum(collision_counts.values())
total_contact = sum(contact_counts.values())
total_all_identified = sum(total_identified.values())

# Calculate overall percentages
overall_collision_pct = (total_collision / total_all_identified) * 100 if total_all_identified > 0 else 0
overall_contact_pct = (total_contact / total_all_identified) * 100 if total_all_identified > 0 else 0

print_and_log("\n" + "="*50)
print_and_log("BINARY CLUSTER LABELING (SIGNIFICANT FEATURES)")
print_and_log("="*50)

# Display overall percentages first
print_and_log("\nOverall sport type distribution:")
print_and_log(f"  - Collision sports: {total_collision} participants ({overall_collision_pct:.1f}% of all identified athletes)")
print_and_log(f"  - Contact sports: {total_contact} participants ({overall_contact_pct:.1f}% of all identified athletes)")
print_and_log(f"  - Total identified: {total_all_identified} participants")

cluster_labels = {}

# Then proceed with the per-cluster breakdowns as before
for cluster in [0, 1]:
    if total_identified[cluster] > 0:
        collision_pct = (collision_counts[cluster] / total_identified[cluster]) * 100
        contact_pct = (contact_counts[cluster] / total_identified[cluster]) * 100
        
        print_and_log(f"\nCluster {cluster} sport type breakdown:")
        print_and_log(f"  - Collision sports: {collision_counts[cluster]} participants ({collision_pct:.1f}%)")
        print_and_log(f"  - Contact sports: {contact_counts[cluster]} participants ({contact_pct:.1f}%)")
        print_and_log(f"  - Total identified: {total_identified[cluster]} participants")
        
        # Add percentage of total for each sport type in this cluster
        collision_of_total_pct = (collision_counts[cluster] / total_collision) * 100 if total_collision > 0 else 0
        contact_of_total_pct = (contact_counts[cluster] / total_contact) * 100 if total_contact > 0 else 0
        print_and_log(f"  - Contains {collision_of_total_pct:.1f}% of all collision sport athletes")
        print_and_log(f"  - Contains {contact_of_total_pct:.1f}% of all contact sport athletes")
        
        # Assign label based on higher percentage
        if collision_pct > contact_pct:
            cluster_labels[cluster] = "Collision"
        else:
            cluster_labels[cluster] = "Contact"
    else:
        cluster_labels[cluster] = "Undetermined"

# Force binary classification if both clusters have same label
if len(set(cluster_labels.values())) == 1 and "Undetermined" not in cluster_labels.values():
    # If both clusters got same label, assign based on relative prevalence
    cluster0_collision_ratio = collision_counts[0] / contact_counts[0] if contact_counts[0] > 0 else float('inf')
    cluster1_collision_ratio = collision_counts[1] / contact_counts[1] if contact_counts[1] > 0 else float('inf')
    
    if cluster0_collision_ratio > cluster1_collision_ratio:
        cluster_labels[0] = "Collision"
        cluster_labels[1] = "Contact"
    else:
        cluster_labels[0] = "Contact"
        cluster_labels[1] = "Collision"
    
    print_and_log("\nBoth clusters had the same label - forcing binary classification based on relative sport type ratios.")

print_and_log("\nFinal binary cluster labels (Significant Features):")
for cluster, label in cluster_labels.items():
    print_and_log(f"Cluster {cluster}: {label}")

# Add the labels to the dataframe
df_clustered["Cluster_2_Label"] = df_clustered["Cluster_2"].map(cluster_labels)

# Save the labeled data
df_clustered.to_csv("Data/Significant_Labeled_Clustered_data_2_clusters.csv", index=False)
print("\nSaved labeled cluster data with sport type labels")

# Create a visual representation of the sport type distribution
plt.figure(figsize=(12, 8))
clusters = [0, 1]
width = 0.35
x = np.arange(len(clusters))

collision_percentages = [collision_counts[cluster]/total_identified[cluster]*100 if total_identified[cluster] > 0 else 0 for cluster in clusters]
contact_percentages = [contact_counts[cluster]/total_identified[cluster]*100 if total_identified[cluster] > 0 else 0 for cluster in clusters]

plt.bar(x - width/2, collision_percentages, width, label='Collision Sports')
plt.bar(x + width/2, contact_percentages, width, label='Contact Sports')

plt.xlabel('Cluster')
plt.ylabel('Percentage of Athletes')
plt.title('Sport Type Distribution by Cluster (Significant Features Only)')
plt.xticks(x, [f"Cluster {i}\n({cluster_labels[i]})" for i in clusters])
plt.ylim(0, 100)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add percentage labels on bars
for i, v in enumerate(collision_percentages):
    plt.text(i - width/2, v + 2, f"{v:.1f}%", ha='center')
for i, v in enumerate(contact_percentages):
    plt.text(i + width/2, v + 2, f"{v:.1f}%", ha='center')

plt.tight_layout()
plt.savefig("Plots/significant_sport_type_distribution.png")
print("\nSaved sport type distribution plot")

# MODIFICATION: Compare results with original clustering
print_and_log("\n" + "="*40)
print_and_log("COMPARISON WITH ORIGINAL CLUSTERING")
print_and_log("="*40)

# Try to load original clustering results
try:
    original_clusters = pd.read_csv("Data/Labeled_Clustered_data_2_clusters.csv")
    
    # Create DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Original_Cluster': original_clusters["Cluster_2"],
        'Significant_Cluster': df_clustered["Cluster_2"]
    })
    
    # Calculate agreement
    agreement = (comparison_df['Original_Cluster'] == comparison_df['Significant_Cluster']).mean() * 100
    
    print_and_log(f"\nAgreement between original and significant feature clustering: {agreement:.2f}%")
    
    # Create contingency table
    contingency = pd.crosstab(
        comparison_df['Original_Cluster'], 
        comparison_df['Significant_Cluster'],
        rownames=['Original'], 
        colnames=['Significant Features']
    )
    
    print_and_log("\nContingency table (Original vs. Significant Features):")
    print_and_log(contingency)
    
    # Visualize the agreement
    plt.figure(figsize=(10, 8))
    sns.heatmap(contingency, annot=True, cmap="YlGnBu", fmt="d")
    plt.title('Agreement Between Original and Significant Features Clustering')
    plt.tight_layout()
    plt.savefig("Plots/original_vs_significant_comparison.png")
    print("\nSaved original vs. significant features comparison plot")
    
except FileNotFoundError:
    print_and_log("\nOriginal clustering results file not found. Skipping comparison.")

# Close the output file
output_file.close()