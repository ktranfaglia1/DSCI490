import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
print(df["Sport_Name"].value_counts())

# Define the selected features
relevant_columns = [
    "Head_Injury_Status", "Concussion_Status", "Lose_Consciousness", 
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
    "Sports_Concussion_Status"
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

# Ensure all columns are numeric before scaling
df_selected = df_selected.apply(pd.to_numeric, errors='coerce')

# Fill missing values
for col in df_selected.columns:
    if col in time_columns:
        df_selected[col] = df_selected[col].fillna(df_selected[col].median())
    else:
        if not df_selected[col].mode().empty:
            df_selected[col] = df_selected[col].fillna(df_selected[col].mode().iloc[0])
        else:
            df_selected[col] = df_selected[col].fillna(0)  # Default to 0 if no mode exists

# Normalize and apply PCA
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=df_selected.columns)
pca = PCA(n_components=10)
df_pca = pd.DataFrame(pca.fit_transform(df_scaled), columns=[f"PC{i+1}" for i in range(10)])

# Apply clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_pca["Cluster"] = kmeans.fit_predict(df_pca)

df_clustered = df.loc[df_selected.index, ["Sport_Name"]].copy()
df_clustered["Cluster"] = df_pca["Cluster"]

# Show results
print("Cluster distribution:\n", df_clustered["Cluster"].value_counts())
print(df_clustered.groupby("Cluster")["Sport_Name"].value_counts())
