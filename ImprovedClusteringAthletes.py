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
output_file = open("./Data/Improved_Clustering_output.txt", "w")

def print_and_log(message):
    print(message)
    output_file.write(str(message) + "\n")

print_and_log("IMPROVED CLUSTERING ANALYSIS\n" + "="*30)

# Load dataset
file_path = "./Data/Labeled_survey_data.csv"
df = pd.read_csv(file_path)

df = df[df['Sports_Status'] == 'Yes']

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

# Define the selected features (keeping the same as original)
relevant_columns = [
    "Bed_Time", "Min_To_Sleep", "Wake_Up", "Sleep_Per_Night", "Cant_Sleep", 
    "Wake_In_Night", "Wake_To_Bathroom", "Breathe_Discomfort", "Snore_Cough", 
    "Too_Cold", "Too_Hot", "Bad_Dreams", "Pain", "Sleep_Quality", "Sleep_Meds", 
    "Staying_Awake_Issues", "Motivation_Issues", "Noise_Concentration_Issues", 
    "Concentration_Issues", "Surrounding_Concentration_Issues", "Good_Music_Concentration", 
    "Concentration_Aware", "Reading_Concentration_Issues", "Trouble_Blocking_Thoughts", 
    "Excitement_Concentration_Issues", "Ignore_Hunger_Concentrating", "Good_Task_Switching", 
    "Long_Time_Focus", "Poor_Listening_Writing", "Quick_Interest", "Easy_Read_Write_On_Phone", 
    "Trouble_Multiple_Conversations", "Trouble_Quick_Creativity", "Good_Interruption_Recovery", 
    "Good_Thought_Recovery", "Good_Task_Alteration", "Poor_Perspective_Thinking", 
    "Sports_Concussion_Status", "Head_Injury_Status", "Concussion_Status", "Lose_Consciousness"
]

# Keep only relevant columns
df_selected = df[relevant_columns].copy()

# Convert binary categorical data (Yes/No) to 0/1
binary_mappings = {"Yes": 1, "No": 0}
for col in ["Head_Injury_Status", "Concussion_Status", "Lose_Consciousness", "Sports_Concussion_Status"]:
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

df_selected[time_columns] = df_selected[time_columns].apply(pd.to_numeric, errors='coerce')

# Mapping ordinal categorical responses
frequency_mappings = {
    "Not during the past month": 0,
    "Less than once a week": 1,
    "Once or twice a week": 2,
    "Three or more times a week": 3
}
ordinal_mappings = {
    "Poor": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5,
    "Almost never": 0, "Sometimes": 1, "Often": 2, "Always": 3
}

for col in ["Cant_Sleep", "Wake_In_Night", "Wake_To_Bathroom", "Breathe_Discomfort",
            "Snore_Cough", "Too_Cold", "Too_Hot", "Bad_Dreams", "Pain",
            "Sleep_Meds", "Staying_Awake_Issues"]:
    df_selected[col] = df_selected[col].map(frequency_mappings)

for col in ["Sleep_Quality", "Motivation_Issues", "Concentration_Issues",
            "Trouble_Blocking_Thoughts", "Good_Task_Switching", "Good_Thought_Recovery",
            "Good_Task_Alteration", "Poor_Perspective_Thinking", "Noise_Concentration_Issues",
            "Surrounding_Concentration_Issues", "Reading_Concentration_Issues", 
            "Excitement_Concentration_Issues", "Ignore_Hunger_Concentrating",
            "Long_Time_Focus", "Poor_Listening_Writing", "Quick_Interest",
            "Easy_Read_Write_On_Phone", "Trouble_Multiple_Conversations", 
            "Trouble_Quick_Creativity", "Good_Interruption_Recovery"]:
    df_selected[col] = df_selected[col].map(ordinal_mappings)

# IMPROVEMENT 1: Create new features
print_and_log("\nCreating composite features for better clustering...")

# 1. Sleep quality index
sleep_quality_cols = ["Cant_Sleep", "Wake_In_Night", "Wake_To_Bathroom", "Breathe_Discomfort",
                      "Snore_Cough", "Too_Cold", "Too_Hot", "Bad_Dreams", "Pain", 
                      "Sleep_Meds", "Sleep_Quality"]
df_selected['Sleep_Problem_Index'] = df_selected[sleep_quality_cols].mean(axis=1)

# 2. Concentration problems index
concentration_cols = ["Noise_Concentration_Issues", "Concentration_Issues", 
                      "Surrounding_Concentration_Issues", "Reading_Concentration_Issues",
                      "Trouble_Blocking_Thoughts", "Excitement_Concentration_Issues"]
df_selected['Concentration_Problem_Index'] = df_selected[concentration_cols].mean(axis=1)

# 3. Task switching ability index
task_switching_cols = ["Good_Task_Switching", "Good_Interruption_Recovery", 
                       "Good_Thought_Recovery", "Good_Task_Alteration"]
df_selected['Task_Switching_Index'] = df_selected[task_switching_cols].mean(axis=1)

# Display summary of the new features
print_and_log("\nNew derived features summary:")
for feature in ['Sleep_Problem_Index', 'Concentration_Problem_Index', 'Task_Switching_Index']:
    print_and_log(f"{feature}: Mean = {df_selected[feature].mean():.2f}, Median = {df_selected[feature].median():.2f}")

# Ensure all columns are numeric before scaling
df_selected = df_selected.apply(pd.to_numeric, errors='coerce')

# Fill missing values with more sophisticated methods
for col in df_selected.columns:
    if df_selected[col].isna().sum() > 0:
        if col in ['Sleep_Problem_Index', 'Concentration_Problem_Index', 'Task_Switching_Index'] + time_columns:
            # For continuous variables, use median
            df_selected[col] = df_selected[col].fillna(df_selected[col].median())
        else:
            # For ordinal/categorical variables, use mode
            if not df_selected[col].mode().empty:
                df_selected[col] = df_selected[col].fillna(df_selected[col].mode().iloc[0])
            else:
                df_selected[col] = df_selected[col].fillna(0)  # Default to 0 if no mode exists

# IMPROVEMENT 2: Use RobustScaler instead of StandardScaler to reduce influence of outliers
print_and_log("\nScaling data using RobustScaler to minimize outlier impact...")
robust_scaler = RobustScaler()
df_scaled = pd.DataFrame(robust_scaler.fit_transform(df_selected), columns=df_selected.columns)

# IMPROVEMENT 3: Determine optimal number of PCs based on explained variance
pca_full = PCA()
pca_full.fit(df_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find number of components that explain at least 90% of variance
n_components = np.argmax(cumulative_variance >= 0.80) + 1
print_and_log(f"\nOptimal number of PCs explaining ≥ 80% variance: {n_components}")

# IMPROVEMENT 4: Apply PCA with optimal components
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

# IMPROVEMENT 5: Visualize feature contributions to principal components
component_loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(n_components)],
    index=df_scaled.columns
)

# Plot heatmap of feature loadings for the first 5 PCs
plt.figure(figsize=(16, 14))
top_features = 15  # Show loadings for top features
loading_abs = component_loadings.abs().sum(axis=1).sort_values(ascending=False)
top_loading_features = loading_abs.head(top_features).index
loadings_heatmap = sns.heatmap(
    component_loadings.loc[top_loading_features, component_loadings.columns[:5]],
    cmap="coolwarm", 
    annot=True, 
    fmt=".2f",
    cbar_kws={"label": "Feature Contribution"}
)
plt.title('Feature Contributions to Principal Components Athletes Only', fontsize=16)
plt.tight_layout()
plt.savefig("Plots/pca_feature_loadings_athletes_only.png")
print("\nSaved PCA feature loadings heatmap")

# IMPROVEMENT 6: Determine optimal number of clusters using multiple metrics
print_and_log("\nDetermining optimal number of clusters...")
k_range = range(2, 6)  # Test 2-5 clusters based on your suggestion
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
ax[0].set_title('Silhouette Score (higher is better) Athletes Only')
ax[0].grid(True)

ax[1].plot(k_range, ch_scores, 'o-')
ax[1].set_xlabel('Number of clusters')
ax[1].set_ylabel('Calinski-Harabasz Score')
ax[1].set_title('Calinski-Harabasz Score (higher is better) Athletes Only')
ax[1].grid(True)

ax[2].plot(k_range, db_scores, 'o-')
ax[2].set_xlabel('Number of clusters')
ax[2].set_ylabel('Davies-Bouldin Score')
ax[2].set_title('Davies-Bouldin Score (lower is better) Athletes Only')
ax[2].grid(True)

plt.tight_layout()
plt.savefig("Plots/improved_cluster_metrics_athletes_only.png")
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
final_k = Counter([best_k_silhouette, best_k_ch, best_k_db]).most_common(1)[0][0]
print_and_log(f"\nSelected number of clusters based on majority vote: {final_k}")

# Perform clustering with optimal k and 2 clusters
for k in [final_k, 2]:
    print_and_log(f"\n{'='*20} ANALYSIS WITH {k} CLUSTERS {'='*20}")
    
    # IMPROVEMENT 7: Apply K-Means with the optimal number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_pca[f"Cluster_{k}"] = kmeans.fit_predict(df_pca)
    
    # Add back Sport_Name and create final dataframe
    df_clustered = df.loc[df_selected.index, ["Sport_Name"]].copy()
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
    cluster_features = pd.concat([df_selected, df_clustered[f"Cluster_{k}"]], axis=1)
    
    # Calculate and display cluster profiles
    print_and_log(f"\nCluster profiles for k={k}:")
    cluster_profiles = cluster_features.groupby(f"Cluster_{k}").mean()
    
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
    
    # IMPROVEMENT 8: Visualization with t-SNE for better cluster separation
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
    plt.title(f't-SNE Visualization of {k} Clusters Athletes Only', fontsize=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title='Cluster', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"Plots/tsne_cluster_{k}_scatter_athletes_only.png")
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
    # Get the current legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Replace them with integer versions
    plt.legend(handles, [str(int(float(label))) for label in labels], title='Cluster', fontsize=10)
    plt.title(f'Athlete Only PCA Visualization of {k} Clusters Athletes Only', fontsize=15)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title='Cluster', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"Plots/atheles_pca_cluster_{k}_scatter_athletes_only.png")
    print(f"\nSaved PCA scatter plot for {k} clusters")
    
    # Visualization 3: Cluster centers heatmap
    plt.figure(figsize=(16, 10))
    centers = kmeans.cluster_centers_
    
    # Create a more informative heatmap with feature importance
    component_names = [f"PC{i+1}" for i in range(min(5, n_components))]  # Top 5 components
    centers_df = pd.DataFrame(centers[:, :len(component_names)], columns=component_names)
    centers_df.index = [f"Cluster {i}" for i in range(k)]
    
    sns.heatmap(
        centers_df,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title(f'K-means Cluster Centers for {k} Clusters (Top 5 PCs) Athletes Only', fontsize=15)
    plt.tight_layout()
    plt.savefig(f"Plots/cluster_{k}_centers_heatmap_athletes_only.png")
    print(f"\nSaved cluster centers heatmap for {k} clusters")
    
    # Visualization 4: Spider plot of cluster profiles for the top features
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
    plt.title(f'Cluster Profiles Comparison for {k} Clusters Athletes Only', size=15)
    plt.tight_layout()
    plt.savefig(f"Plots/cluster_{k}_profiles_spider_athletes_only.png")
    print(f"\nSaved cluster profiles spider plot for {k} clusters")
    
    # Save the clustered data to CSV
    df_clustered.to_csv(f"Data/Clustered_data_results_{k}_clusters.csv", index=False)
    print(f"\nSaved clustered data for {k} clusters")

# IMPROVEMENT 9: Compare with hierarchical clustering for validation
print_and_log("\n" + "-"*60)
print_and_log("\nComparing with Hierarchical Clustering for validation...")

# Apply Agglomerative Clustering with optimal k
agg_clustering = AgglomerativeClustering(n_clusters=final_k)
df_pca[f"Hierarchical_Cluster"] = agg_clustering.fit_predict(df_pca.drop(columns=[f"Cluster_{final_k}", f"Cluster_2"]))

# Calculate agreement between K-means and hierarchical clustering
kmeans_labels = df_pca[f"Cluster_{final_k}"]
hierarchical_labels = df_pca["Hierarchical_Cluster"]

# Create a contingency table
contingency = pd.crosstab(kmeans_labels, hierarchical_labels)
print_and_log("\nContingency table (K-means vs Hierarchical):")
print_and_log(contingency)

# Visualize the agreement
plt.figure(figsize=(10, 8))
sns.heatmap(contingency, annot=True, cmap="YlGnBu", fmt="d")
plt.xlabel('Hierarchical Clustering Labels')
plt.ylabel('K-means Clustering Labels')
plt.title('Agreement Between K-means and Hierarchical Clustering Athletes Only')
plt.tight_layout()
plt.savefig("Plots/clustering_methods_comparison_athletes_only.png")
print("\nSaved clustering methods comparison plot")

# IMPROVEMENT 10: Final summary and recommendations
print_and_log("\n" + "="*40)
print_and_log("CLUSTERING ANALYSIS SUMMARY")
print_and_log("="*40)
print_and_log(f"Optimal number of clusters based on majority vote: {final_k}")
print_and_log(f"Requested number of clusters: 2")
print_and_log("\nKey findings:")

# Extract the most important distinguishing features
for k in [final_k, 2]:
    print_and_log(f"\nFor {k} clusters:")
    # Re-extract the cluster profiles
    cluster_features = pd.concat([df_selected, df_clustered[f"Cluster_{k}"]], axis=1)
    cluster_profiles = cluster_features.groupby(f"Cluster_{k}").mean()
    
    # Find distinguishing features
    for i in range(k):
        # Calculate how different this cluster is from others for each feature
        diff_features = {}
        for feature in top_feature_names:
            # Get the mean for this cluster
            cluster_mean = cluster_profiles.loc[i, feature]
            # Get the mean for all other clusters
            other_mean = cluster_profiles.drop(i).mean()[feature]
            # Calculate difference
            diff = cluster_mean - other_mean
            diff_features[feature] = (diff, cluster_mean, other_mean)
        
        # Sort by absolute difference
        sorted_diffs = sorted(diff_features.items(), key=lambda x: abs(x[1][0]), reverse=True)
        
        cluster_size = (df_clustered[f"Cluster_{k}"] == i).sum()
        cluster_percentage = cluster_size / len(df_clustered) * 100
        
        print_and_log(f"Cluster {i} ({cluster_size} members, {cluster_percentage:.1f}%):")
        
        # Get top 5 distinguishing features for this cluster
        for j, (feature, (diff, cluster_mean, other_mean)) in enumerate(sorted_diffs[:5]):
            direction = "higher" if diff > 0 else "lower"
            print_and_log(f"  - {feature}: {direction} than other clusters ({cluster_mean:.2f} vs {other_mean:.2f})")
        
        # Get all sports in this cluster with their counts and percentages
        sports_in_cluster = df_clustered[df_clustered[f"Cluster_{k}"] == i]["Sport_Name"].value_counts()
        all_sports_counts = df_clustered["Sport_Name"].value_counts()

        if not sports_in_cluster.empty:
            print_and_log("  Sports distribution:")
            for sport, count in sports_in_cluster.items():
                if pd.notna(sport):
                    # Calculate percentage of this sport's participants that are in this cluster
                    total_sport_count = all_sports_counts.get(sport, 0)
                    percentage = (count / total_sport_count) * 100 if total_sport_count > 0 else 0
                    print_and_log(f"    - {sport}: {count} participants ({percentage:.1f}% of all {sport} participants)")

print_and_log("\n" + "="*40)
print_and_log("Binary Cluster Labeling")
print_and_log("="*40)

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
    sports_in_cluster = df_clustered[df_clustered[f"Cluster_2"] == cluster]["Sport_Name"].value_counts()
    cluster_total = df_clustered[df_clustered[f"Cluster_2"] == cluster].shape[0]
    
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
    
    # Check for unidentified sports and report them
    if unidentified_counts[cluster] > 0:
        print_and_log(f"\nWarning: {unidentified_counts[cluster]} participants in Cluster {cluster} have unidentified sports.")

# Calculate percentages and assign labels
cluster_labels = {}
for cluster in [0, 1]:
    if total_identified[cluster] > 0:
        collision_pct = (collision_counts[cluster] / total_identified[cluster]) * 100
        contact_pct = (contact_counts[cluster] / total_identified[cluster]) * 100
        
        print_and_log(f"\nCluster {cluster} sport type breakdown:")
        print_and_log(f"  - Collision sports: {collision_counts[cluster]} participants ({collision_pct:.1f}%)")
        print_and_log(f"  - Contact sports: {contact_counts[cluster]} participants ({contact_pct:.1f}%)")
        print_and_log(f"  - Total identified: {total_identified[cluster]} participants")
        
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

print_and_log("\nFinal binary cluster labels:")
for cluster, label in cluster_labels.items():
    print_and_log(f"Cluster {cluster}: {label}")

# Add the labels to the dataframe
df_clustered[f"Cluster_2_Label"] = df_clustered[f"Cluster_2"].map(cluster_labels)

# Save the labeled data
df_clustered.to_csv(f"Data/Labeled_Clustered_data_2_clusters.csv", index=False)
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
plt.title('Sport Type Distribution by Cluster Athletes Only')
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
plt.savefig("Plots/sport_type_distribution_athletes_only.png")
print("\nSaved sport type distribution plot")

# Close the output file
print("\nAll analysis complete. Results saved to files.")
output_file.close()