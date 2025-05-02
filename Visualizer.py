#  Author: Kyle Tranfaglia
#  Title: DSCI490 - theDataFixer
#  Last updated:  02/24/25
#  Description: This program is a visualization tool to generate plots relating various features
import pandas as pd
import matplotlib.pyplot as plt
import re


def load_and_clean_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Extract numeric part from Sleep_Per_Night and convert to float
    df['Sleep_Per_Night_Clean'] = df['Sleep_Per_Night'].astype(str).apply(
        lambda x: float(re.search(r'\d+(\.\d+)?', x).group()) if re.search(r'\d+(\.\d+)?', x) else None
    )

    # Standardize 'Yes' labels in Sports_Concussion_Status
    df['Sports_Concussion_Status'] = df['Sports_Concussion_Status'].astype(str).apply(
        lambda x: 'Yes' if x.startswith('Yes') else 'No'
    )

    return df


def plot_sports_concussion_distribution(df):
    plt.figure(figsize=(6, 4))
    df['Sports_Concussion_Status'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title("Frequency of Sports-Related Concussions (General Population)")
    plt.xlabel("Sports Concussion Status")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.savefig("./Plots/sports_concussion_distribution.png")
    plt.close()


def plot_sports_concussion_distribution_for_athletes(df):
    """Plots sports-related concussion distribution only for people who play sports."""
    athletes_df = df[df['Sports_Status'].str.lower() == 'yes']
    
    plt.figure(figsize=(6, 4))
    athletes_df['Sports_Concussion_Status'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title("Frequency of Sports-Related Concussions Among Athletes")
    plt.xlabel("Sports Concussion Status")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.savefig("./Plots/sports_concussion_distribution_athletes.png")
    plt.close()


def plot_concussion_distribution_for_non_athletes(df):
    """Plots sports-related concussion distribution only for people who do not play sports."""
    athletes_df = df[df['Sports_Status'].str.lower() == 'no']
    
    plt.figure(figsize=(6, 4))
    athletes_df['Concussion_Status'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title("Frequency of Concussions Among Non-Athletes")
    plt.xlabel("Concussion Status")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.savefig("./Plots/sports_concussion_distribution_non_athletes.png")
    plt.close()


def plot_general_concussion_distribution(df):
    plt.figure(figsize=(6, 4))
    df['Concussion_Status'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title("Frequency of General Concussions")
    plt.xlabel("Concussion Status")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.savefig("./Plots/general_concussion_distribution.png")
    plt.close()


def plot_combined_concussion_distribution(df):
    plt.figure(figsize=(8, 5))

    athletes_df = df[df['Sports_Status'].str.lower() == 'yes']

    # Count values for all categories
    sports_counts = df['Sports_Concussion_Status'].value_counts()
    general_counts = df['Concussion_Status'].value_counts()
    athelete_counts = athletes_df['Sports_Concussion_Status'].value_counts()

    # Create a combined DataFrame for plotting
    combined_df = pd.DataFrame({
        'Sports-Related': sports_counts,
        'General': general_counts,
        'Among Athletes': athelete_counts
    }).T.fillna(0)

    # Plot
    combined_df.plot(kind='bar', color=['skyblue', 'salmon'], width=0.7)
    plt.title("Combined Concussion Distributions")
    plt.xlabel("Concussion Type")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.legend(title="Status")
    plt.savefig("./Plots/combined_concussion_distribution.png")
    plt.close()


def remove_sleep_outliers(df):
    """Removes upper-bound sleep outliers using IQR method."""
    sleep_data = df['Sleep_Per_Night_Clean'].dropna()

    # Compute IQR
    Q1 = sleep_data.quantile(0.25)
    Q3 = sleep_data.quantile(0.75)
    IQR = Q3 - Q1

    # Define upper bound
    upper_bound = Q3 + 1.5 * IQR

    max_sleep = max(upper_bound, 16)

    # Filter out extreme values
    df_filtered = df[df['Sleep_Per_Night_Clean'] <= max_sleep]
    
    return df_filtered


def plot_sleep_distribution(df):
    """Plots sleep distribution after removing outliers."""
    df_filtered = remove_sleep_outliers(df)

    plt.figure(figsize=(8, 5))
    df_filtered['Sleep_Per_Night_Clean'].dropna().hist(bins=10, color='purple', alpha=0.7)
    plt.title("Distribution of Sleep Hours Per Night")
    plt.xlabel("Hours of Sleep")
    plt.ylabel("Frequency")
    plt.grid(False)
    plt.savefig("./Plots/sleep_distribution.png")
    plt.close()


def plot_avg_sleep_by_concussion_status(df):
    """Plots average sleep per night by concussion status after removing outliers."""
    df_filtered = remove_sleep_outliers(df)

    # Compute mean sleep hours per night for each concussion status
    avg_sleep = df_filtered.groupby('Concussion_Status')['Sleep_Per_Night_Clean'].mean()

    # Plot bar chart
    plt.figure(figsize=(6, 4))
    avg_sleep.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title("Average Sleep Hours per Night by Concussion Status")
    plt.xlabel("Sports Concussion Status")
    plt.ylabel("Average Hours of Sleep")
    plt.xticks(rotation=0)
    plt.ylim(0, df_filtered['Sleep_Per_Night_Clean'].max() + 1)
    plt.savefig("./Plots/avg_sleep_by_concussion_status.png")
    plt.close()


def plot_concentration_issues_general(df):
    """Plots percentage of concentration issues by concussion status with count labels."""
    # Create a copy of the dataframe to avoid modifying the original
    df_plot = df.copy()
    
    # Define the desired order
    desired_order = ["Almost never", "Sometimes", "Often", "Always"]
    
    # Convert the Concentration_Issues column to a Categorical type with the specific order
    df_plot['Concentration_Issues'] = pd.Categorical(
        df_plot['Concentration_Issues'], 
        categories=desired_order, 
        ordered=True
    )
    
    # Compute actual counts
    count_data = pd.crosstab(df_plot['Concentration_Issues'], df_plot['Concussion_Status'])

    # Ensure column order is No → Yes
    count_data = count_data[['No', 'Yes']]

    # Normalize to get percentages
    concentration_percentage = count_data.div(count_data.sum(axis=0), axis=1) * 100

    concentration_percentage.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'], width=0.7)

    plt.title("Percentage of Concentration Issues by Concussion Status")
    plt.xlabel("Concentration Issues Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.legend(title="Concussion Status")
    plt.savefig("./Plots/concentration_issues_general.png")
    plt.close()


def plot_concentration_issues_for_athletes(df):
    """Plots concentration issue percentages for athletes only, comparing those with and without a concussion."""
    # Create a copy of the dataframe to avoid modifying the original
    df_plot = df.copy()
    
    # Define the desired order
    desired_order = ["Almost never", "Sometimes", "Often", "Always"]
    
    # Convert the Concentration_Issues column to a Categorical type with the specific order
    df_plot['Concentration_Issues'] = pd.Categorical(
        df_plot['Concentration_Issues'], 
        categories=desired_order, 
        ordered=True
    )
    
    athletes_df = df_plot[df_plot['Sports_Status'].str.lower() == 'yes']

    # Compute actual counts
    count_data = pd.crosstab(athletes_df['Concentration_Issues'], athletes_df['Concussion_Status'])

    # Ensure column order is No → Yes
    count_data = count_data[['No', 'Yes']]

    # Normalize to get percentages
    concentration_percentage = count_data.div(count_data.sum(axis=0), axis=1) * 100

    concentration_percentage.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'], width=0.7)

    plt.title("Concentration Issues for Athletes by Concussion Status")
    plt.xlabel("Concentration Issues Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.legend(title="Concussion Status")
    plt.savefig("./Plots/concentration_issues_athletes.png")
    plt.close()


def plot_concentration_issues_for_non_athletes(df):
    """Plots concentration issue percentages for non-athletes only, comparing those with and without a concussion."""
    # Create a copy of the dataframe to avoid modifying the original
    df_plot = df.copy()
    
    # Define the desired order
    desired_order = ["Almost never", "Sometimes", "Often", "Always"]
    
    # Convert the Concentration_Issues column to a Categorical type with the specific order
    df_plot['Concentration_Issues'] = pd.Categorical(
        df_plot['Concentration_Issues'], 
        categories=desired_order, 
        ordered=True
    )
    
    non_athletes_df = df_plot[df_plot['Sports_Status'].str.lower() == 'no']

    # Compute actual counts
    count_data = pd.crosstab(non_athletes_df['Concentration_Issues'], non_athletes_df['Concussion_Status'])

    # Ensure column order is No → Yes
    count_data = count_data[['No', 'Yes']]

    # Normalize to get percentages
    concentration_percentage = count_data.div(count_data.sum(axis=0), axis=1) * 100

    concentration_percentage.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'], width=0.7)

    plt.title("Concentration Issues for Non-Athletes by Concussion Status")
    plt.xlabel("Concentration Issues Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.legend(title="Concussion Status")
    plt.savefig("./Plots/concentration_issues_non_athletes.png")
    plt.close()


def plot_motivation_issues_general(df):
    """Plots the distribution of Motivation_Issues responses per concussion status."""
    count_data = pd.crosstab(df['Motivation_Issues'], df['Concussion_Status'])
    count_data = count_data[['No', 'Yes']]  # Ensure order is No → Yes
    
    # Normalize to get percentages
    motivation_percentage = count_data.div(count_data.sum(axis=0), axis=1) * 100
    
    motivation_percentage.plot(kind='bar', figsize=(10, 5), color=['skyblue', 'salmon'], width=0.7)
    
    plt.title("Distribution of Motivation Issues by Concussion Status")
    plt.xlabel("Motivation Issues Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.legend(title="Concussion Status")
    plt.savefig("./Plots/motivation_issues_general.png")
    plt.close()


def plot_motivation_issues_for_athletes(df):
    """Plots motivation issue percentages for athletes only, comparing those with and without a concussion."""
    athletes_df = df[df['Sports_Status'].str.lower() == 'yes']

    count_data = pd.crosstab(athletes_df['Motivation_Issues'], athletes_df['Concussion_Status'])
    count_data = count_data[['No', 'Yes']]  # Ensure order is No → Yes
    
    # Normalize to get percentages
    motivation_percentage = count_data.div(count_data.sum(axis=0), axis=1) * 100
    
    motivation_percentage.plot(kind='bar', figsize=(10, 5), color=['skyblue', 'salmon'], width=0.7)
    
    plt.title("Distribution of Motivation Issues For Athletes by Concussion Status")
    plt.xlabel("Motivation Issues Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.legend(title="Concussion Status")
    plt.savefig("./Plots/motivation_issues_for_athletes.png")
    plt.close()


def plot_motivation_issues_for_non_athletes(df):
    """Plots motivation issue percentages for non-athletes only, comparing those with and without a concussion."""
    non_athletes_df = df[df['Sports_Status'].str.lower() == 'no']

    count_data = pd.crosstab(non_athletes_df['Motivation_Issues'], non_athletes_df['Concussion_Status'])
    count_data = count_data[['No', 'Yes']]  # Ensure order is No → Yes
    
    # Normalize to get percentages
    motivation_percentage = count_data.div(count_data.sum(axis=0), axis=1) * 100
    
    motivation_percentage.plot(kind='bar', figsize=(10, 5), color=['skyblue', 'salmon'], width=0.7)
    
    plt.title("Distribution of Motivation Issues for Non-Athletes by Concussion Status")
    plt.xlabel("Motivation Issues Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.legend(title="Concussion Status")
    plt.savefig("./Plots/motivation_issues_for_non_athletes.png")
    plt.close()


def plot_pain_issues_general(df):
    """Plots the distribution of Pain responses per concussion status."""
    count_data = pd.crosstab(df['Pain'], df['Concussion_Status'])
    count_data = count_data[['No', 'Yes']]  # Ensure order is No → Yes
    
    # Normalize to get percentages
    motivation_percentage = count_data.div(count_data.sum(axis=0), axis=1) * 100
    
    motivation_percentage.plot(kind='bar', figsize=(10, 5), color=['skyblue', 'salmon'], width=0.7)
    
    plt.title("Distribution of Pain Issues by Concussion Status")
    plt.xlabel("Pain Issues Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.legend(title="Concussion Status")
    plt.savefig("./Plots/pain_issues_general.png")
    plt.close()


def plot_pain_issues_for_athletes(df):
    """Plots pain issue percentages for athletes only, comparing those with and without a concussion."""
    athletes_df = df[df['Sports_Status'].str.lower() == 'yes']

    count_data = pd.crosstab(athletes_df['Pain'], athletes_df['Concussion_Status'])
    count_data = count_data[['No', 'Yes']]  # Ensure order is No → Yes
    
    # Normalize to get percentages
    motivation_percentage = count_data.div(count_data.sum(axis=0), axis=1) * 100
    
    motivation_percentage.plot(kind='bar', figsize=(10, 5), color=['skyblue', 'salmon'], width=0.7)
    
    plt.title("Distribution of Pain Issues for Athletes by Concussion Status")
    plt.xlabel("Pain Issues Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.legend(title="Concussion Status")
    plt.savefig("./Plots/pain_issues_for_athletes.png")
    plt.close()


def plot_pain_issues_for_non_athletes(df):
    """Plots pain issue percentages for non-athletes only, comparing those with and without a concussion."""
    non_athletes_df = df[df['Sports_Status'].str.lower() == 'no']

    count_data = pd.crosstab(non_athletes_df['Pain'], non_athletes_df['Concussion_Status'])
    count_data = count_data[['No', 'Yes']]  # Ensure order is No → Yes
    
    # Normalize to get percentages
    motivation_percentage = count_data.div(count_data.sum(axis=0), axis=1) * 100
    
    motivation_percentage.plot(kind='bar', figsize=(10, 5), color=['skyblue', 'salmon'], width=0.7)
    
    plt.title("Distribution of Pain Issues for Non-Athletes by Concussion Status")
    plt.xlabel("Pain Issues Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.legend(title="Concussion Status")
    plt.savefig("./Plots/pain_issues_for_non_athletes.png")
    plt.close()


def plot_sleep_quality_general(df):
    """Plots the distribution of sleep quality responses per concussion status."""
    count_data = pd.crosstab(df['Sleep_Quality'], df['Concussion_Status'])
    count_data = count_data[['No', 'Yes']]  # Ensure order is No → Yes
    
    # Normalize to get percentages
    motivation_percentage = count_data.div(count_data.sum(axis=0), axis=1) * 100
    
    # Create a mapping for better labels
    sleep_quality_labels = {
        0.0: "Very bad", 
        1.0: "Fairly bad", 
        2.0: "Fairly good", 
        3.0: "Very good"
    }
    
    # Rename the index with descriptive labels
    motivation_percentage.index = motivation_percentage.index.map(sleep_quality_labels)
    
    motivation_percentage.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'], width=0.7)
    
    plt.title("Distribution of Sleep Quality by Concussion Status")
    plt.xlabel("Sleep Quality Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.legend(title="Concussion Status")
    plt.savefig("./Plots/sleep_quality_general.png")
    plt.close()


def plot_sleep_quality_for_athletes(df):
    """Plots sleep quality percentages for athletes only, comparing those with and without a concussion."""
    athletes_df = df[df['Sports_Status'].str.lower() == 'yes']

    count_data = pd.crosstab(athletes_df['Sleep_Quality'], athletes_df['Concussion_Status'])
    count_data = count_data[['No', 'Yes']]  # Ensure order is No → Yes
    
    # Normalize to get percentages
    motivation_percentage = count_data.div(count_data.sum(axis=0), axis=1) * 100

    # Create a mapping for better labels
    sleep_quality_labels = {
        0.0: "Very bad", 
        1.0: "Fairly bad", 
        2.0: "Fairly good", 
        3.0: "Very good"
    }
    
    # Rename the index with descriptive labels
    motivation_percentage.index = motivation_percentage.index.map(sleep_quality_labels)
    
    motivation_percentage.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'], width=0.7)
    
    plt.title("Distribution of Sleep Quality for Athletes by Concussion Status")
    plt.xlabel("Sleep Quality Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.legend(title="Concussion Status")
    plt.savefig("./Plots/sleep_quality_for_athletes.png")
    plt.close()


def plot_sleep_quality_for_non_athletes(df):
    """Plots sleep quality percentages for non-athletes only, comparing those with and without a concussion."""
    non_athletes_df = df[df['Sports_Status'].str.lower() == 'no']

    count_data = pd.crosstab(non_athletes_df['Sleep_Quality'], non_athletes_df['Concussion_Status'])
    count_data = count_data[['No', 'Yes']]  # Ensure order is No → Yes
    
    # Normalize to get percentages
    motivation_percentage = count_data.div(count_data.sum(axis=0), axis=1) * 100

    # Create a mapping for better labels
    sleep_quality_labels = {
        0.0: "Very bad", 
        1.0: "Fairly bad", 
        2.0: "Fairly good", 
        3.0: "Very good"
    }
    
    # Rename the index with descriptive labels
    motivation_percentage.index = motivation_percentage.index.map(sleep_quality_labels)
    
    motivation_percentage.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'], width=0.7)
    
    plt.title("Distribution of Sleep Quality for Non-Athletes by Concussion Status")
    plt.xlabel("Sleep Quality Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.legend(title="Concussion Status")
    plt.savefig("./Plots/sleep_quality_for_non_athletes.png")
    plt.close()


def main():
    file_path = "./Data/Labeled_survey_data.csv"
    df = load_and_clean_data(file_path)

    plot_sports_concussion_distribution(df)
    plot_sports_concussion_distribution_for_athletes(df)
    plot_concussion_distribution_for_non_athletes(df)
    plot_general_concussion_distribution(df)
    plot_combined_concussion_distribution(df)
    plot_sleep_distribution(df)
    plot_avg_sleep_by_concussion_status(df)
    plot_concentration_issues_general(df)
    plot_concentration_issues_for_athletes(df)
    plot_concentration_issues_for_non_athletes(df)
    plot_motivation_issues_general(df)
    plot_motivation_issues_for_athletes(df)
    plot_motivation_issues_for_non_athletes(df)
    plot_pain_issues_general(df)
    plot_pain_issues_for_athletes(df)
    plot_pain_issues_for_non_athletes(df)
    plot_sleep_quality_general(df)
    plot_sleep_quality_for_athletes(df)
    plot_sleep_quality_for_non_athletes(df)


if __name__ == "__main__":
    main()

