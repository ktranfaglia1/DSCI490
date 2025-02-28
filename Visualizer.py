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
    # Compute actual counts
    count_data = pd.crosstab(df['Concentration_Issues'], df['Concussion_Status'])

    # Ensure column order is No → Yes
    count_data = count_data[['No', 'Yes']]

    # Compute the total number of respondents for No and Yes concussion groups
    total_no = count_data['No'].sum()
    total_yes = count_data['Yes'].sum()

    # Normalize to get percentages
    concentration_percentage = count_data.div(count_data.sum(axis=0), axis=1) * 100

    ax = concentration_percentage.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'], width=0.7)

    # Adding actual raw count labels (calculated from percentages)
    for i, p in enumerate(ax.patches):
        col_idx = i % 2  # Alternates between No and Yes
        row_idx = i // 2  # Each category (Concentration Issues)

        # Determine the correct total population for this bar (No or Yes)
        total_population = total_no if col_idx == 0 else total_yes

        # Convert percentage to actual count
        count_value = round((p.get_height() / 100) * total_population)

        # Display the count
        ax.annotate(f"{count_value}", (p.get_x() + p.get_width() / 2, p.get_height() + 2),
                    ha='center', va='bottom', fontsize=10)

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
    athletes_df = df[df['Sports_Status'].str.lower() == 'yes']

    # Compute actual counts
    count_data = pd.crosstab(athletes_df['Concentration_Issues'], athletes_df['Concussion_Status'])

    # Ensure column order is No → Yes
    count_data = count_data[['No', 'Yes']]

    # Compute the total number of athletes for No and Yes concussion groups
    total_no = count_data['No'].sum()
    total_yes = count_data['Yes'].sum()

    # Normalize to get percentages
    concentration_percentage = count_data.div(count_data.sum(axis=0), axis=1) * 100

    ax = concentration_percentage.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'], width=0.7)

    # Adding actual raw count labels (calculated from percentages)
    for i, p in enumerate(ax.patches):
        col_idx = i % 2  # Alternates between No and Yes
        row_idx = i // 2  # Each category (Concentration Issues)

        # Determine the correct total population for this bar (No or Yes)
        total_population = total_no if col_idx == 0 else total_yes

        # Convert percentage to actual count
        count_value = round((p.get_height() / 100) * total_population)

        # Display the count
        ax.annotate(f"{count_value}", (p.get_x() + p.get_width() / 2, p.get_height() + 2),
                    ha='center', va='bottom', fontsize=10)

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
    non_athletes_df = df[df['Sports_Status'].str.lower() == 'no']

    # Compute actual counts
    count_data = pd.crosstab(non_athletes_df['Concentration_Issues'], non_athletes_df['Concussion_Status'])

    # Ensure column order is No → Yes
    count_data = count_data[['No', 'Yes']]

    # Compute the total number of non-athletes for No and Yes concussion groups
    total_no = count_data['No'].sum()
    total_yes = count_data['Yes'].sum()

    # Normalize to get percentages
    concentration_percentage = count_data.div(count_data.sum(axis=0), axis=1) * 100

    ax = concentration_percentage.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'], width=0.7)

    # Adding actual raw count labels (calculated from percentages)
    for i, p in enumerate(ax.patches):
        col_idx = i % 2  # Alternates between No and Yes
        row_idx = i // 2  # Each category (Concentration Issues)

        # Determine the correct total population for this bar (No or Yes)
        total_population = total_no if col_idx == 0 else total_yes

        # Convert percentage to actual count
        count_value = round((p.get_height() / 100) * total_population)

        # Display the count
        ax.annotate(f"{count_value}", (p.get_x() + p.get_width() / 2, p.get_height() + 2),
                    ha='center', va='bottom', fontsize=10)

    plt.title("Concentration Issues for Non-Athletes by Concussion Status")
    plt.xlabel("Concentration Issues Response")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
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
    plot_concentration_issues_general(df)
    plot_concentration_issues_for_athletes(df)
    plot_concentration_issues_for_non_athletes(df)


if __name__ == "__main__":
    main()

