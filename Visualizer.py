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
    plt.title("Frequency of Sports-Related Concussions")
    plt.xlabel("Sports Concussion Status")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.savefig("./Plots/sports_concussion_distribution.png")
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

    # Count values for both categories
    sports_counts = df['Sports_Concussion_Status'].value_counts()
    general_counts = df['Concussion_Status'].value_counts()

    # Create a combined DataFrame for plotting
    combined_df = pd.DataFrame({
        'Sports-Related Concussions': sports_counts,
        'General Concussions': general_counts
    }).T.fillna(0)

    # Plot
    combined_df.plot(kind='bar', color=['skyblue', 'salmon'], width=0.7)
    plt.title("Comparison of Sports and General Concussions")
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


def main():
    file_path = "./Data/Labeled_survey_data.csv"
    df = load_and_clean_data(file_path)

    plot_combined_concussion_distribution(df)
    plot_sports_concussion_distribution(df)
    plot_general_concussion_distribution(df)
    plot_sleep_distribution(df)


if __name__ == "__main__":
    main()

