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


def plot_sleep_distribution(df):
    plt.figure(figsize=(8, 5))
    df['Sleep_Per_Night_Clean'].dropna().hist(bins=10, color='purple', alpha=0.7)
    plt.title("Distribution of Sleep Hours Per Night")
    plt.xlabel("Hours of Sleep")
    plt.ylabel("Frequency")
    plt.savefig("./Plots/sleep_distribution.png")
    plt.close()


def plot_avg_sleep_by_concussion_status(df):
    # Compute mean sleep hours per night for each concussion status
    avg_sleep = df.groupby('Concussion_Status')['Sleep_Per_Night_Clean'].mean()

    # Plot bar chart
    plt.figure(figsize=(6, 4))
    avg_sleep.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title("Average Sleep Hours per Night by Concussion Status")
    plt.xlabel("Sports Concussion Status")
    plt.ylabel("Average Hours of Sleep")
    plt.xticks(rotation=0)
    plt.ylim(0, df['Sleep_Per_Night_Clean'].max() + 1)
    plt.savefig("./Plots/avg_sleep_by_concussion_status.png")
    plt.close()


def plot_concentration_issues_percentage(df):
    # Compute percentage distribution within each concussion status
    concentration_percentage = pd.crosstab(df['Concentration_Issues'], df['Concussion_Status'], normalize='columns') * 100

    # Plot grouped bar chart
    concentration_percentage.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'], width=0.7)
    plt.title("Percentage of Concentration Issues by Concussion Status")
    plt.xlabel("Concentration Issues Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)  # Rotate labels for readability
    plt.ylim(0, 100)  # Ensure y-axis represents 0-100%
    plt.legend(title="Sports Concussion Status")
    plt.savefig("./Plots/concentration_issues_percentage_by_concussion_status.png")
    plt.close()


def plot_concentration_issues_for_athletes(df):
    """Plots concentration issue percentages for athletes only, comparing those with and without a concussion."""
    athletes_df = df[df['Sports_Status'].str.lower() == 'yes']
    
    concentration_percentage = pd.crosstab(athletes_df['Concentration_Issues'], athletes_df['Concussion_Status'], normalize='columns') * 100

    plt.figure(figsize=(8, 5))
    concentration_percentage.plot(kind='bar', color=['skyblue', 'salmon'], width=0.7)
    plt.title("Concentration Issues for Athletes by Concussion Status")
    plt.xlabel("Concentration Issues Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0, ha='right')
    plt.ylim(0, 100)
    plt.legend(title="Concussion Status")
    plt.savefig("./Plots/concentration_issues_athletes.png")
    plt.close()


def plot_concentration_issues_for_non_athletes(df):
    """Plots concentration issue percentages for non-athletes only, comparing those with and without a concussion."""
    non_athletes_df = df[df['Sports_Status'].str.lower() == 'no']
    
    concentration_percentage = pd.crosstab(non_athletes_df['Concentration_Issues'], non_athletes_df['Concussion_Status'], normalize='columns') * 100

    plt.figure(figsize=(8, 5))
    concentration_percentage.plot(kind='bar', color=['skyblue', 'salmon'], width=0.7)
    plt.title("Concentration Issues for Non-Athletes by Concussion Status")
    plt.xlabel("Concentration Issues Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0, ha='right')
    plt.ylim(0, 100)
    plt.legend(title="Concussion Status")
    plt.savefig("./Plots/concentration_issues_non_athletes.png")
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
    plot_concentration_issues_percentage(df)
    plot_concentration_issues_for_athletes(df)
    plot_concentration_issues_for_non_athletes(df)


if __name__ == "__main__":
    main()

