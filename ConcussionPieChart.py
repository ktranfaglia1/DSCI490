import matplotlib.pyplot as plt
import pandas as pd
import json
import regex as re

initial_data = pd.read_csv("./Data/CompleteData.csv")


def addSports(dataframe):
    sports = []
    for sportJson in dataframe:
        # Ensure sportJson is a valid string and not empty or None
        if isinstance(sportJson, str) and sportJson.strip():
            try:
                # Try parsing the JSON string
                sport_data = json.loads(sportJson)
                for sport in sport_data:
                    # Check if "Concussions" is an integer and append accordingly
                    if isinstance(sport.get("Concussions"), int):
                        sports.extend([sport["Sport"]] * sport["Concussions"])
                    else:
                        sports.append(sport["Sport"])
            except json.JSONDecodeError:
                # Skip invalid JSON entries
                wa = 1
    return sports


sports = pd.Series(addSports(initial_data["Q78_1_TEXT"]))
sports = sports.str.strip().str.lower()


def set_regex(regex, set_value, element):
    if re.search(regex, element):
        return set_value
    else:
        return element


sports = sports.apply(lambda x: set_regex(".*cross.*country.*", "cross country", x))
# sports = sports.apply(lambda x: set_regex(".*hockey.*", "hockey", x))
sports = sports.apply(lambda x: set_regex("ice hockey", "hockey", x))
sports = sports.apply(lambda x: set_regex(".*track.*", "track", x))
sports = sports.apply(lambda x: set_regex(".*cheer.*", "cheer", x))
sports = sports.apply(lambda x: set_regex(".*footbal.*", "football", x))
sports = sports.apply(lambda x: set_regex(".*swim.*", "swim", x))
sports = sports.apply(lambda x: set_regex(".*basket.*ball.*", "swim", x))

sports = sports[sports != "nan"]

print(sports.unique())
# Get the value counts for the pie chart
sport_counts = sports.value_counts()

# Identify sports with counts less than 3
small_sports = sport_counts[sport_counts < 2].index

# Merge those sports into 'Other'
# sports = sports.apply(lambda x: "Other" if x in small_sports else x)

sport_counts = sports.value_counts()


# Plotting the Pie Chart
sport_counts.plot(kind="bar")  # Add a border to bars

plt.title("Distribution of Concussions", fontsize=15)  # More prominent title
plt.xlabel("Sport", fontsize=12)  # X-axis label
plt.ylabel("Number of Concussions", fontsize=12)  # Y-axis label
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
plt.tight_layout()  # Adjust layout to prevent cutting off labels

# plt.show()
plt.savefig("./graphs/Distribution_Of_Sports.png", dpi=300)  # Higher resolution
