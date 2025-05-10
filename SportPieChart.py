import matplotlib.pyplot as plt
import pandas as pd
import json
import regex as re
import numpy as np

initial_data = pd.read_csv("./Data/Complete_data.csv")


"""
    Merge Ice Hockey & Hockey #DONE
    Leave rest unmerged #DONE
    Barchart instead of pie chart 
    Clustering based on attention and sleep speartely
    summing together 
"""


def addSports(dataframe):
    sports = []
    for sportJson in dataframe:
        if re.search(r"\[.*\]", sportJson):
            for sport in json.loads(sportJson):
                sports.append(sport["Sport"])
        else:
            sports.append("None")
    return sports


sports = pd.Series(addSports(initial_data["Q77"]))
print(sports)
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

#sports = sports[sports != "nan"]

print(sports.unique())
# Get the value counts for the pie chart
sport_counts = sports.value_counts()

# Identify sports with counts less than 3
small_sports = sport_counts[sport_counts < 3].index
print(sports)
# Merge those sports into 'Other'
sports = sports.apply(lambda x: "Other" if x in small_sports else x)

sport_counts = sports.value_counts()

# Plotting the Pie Chart
sport_counts.plot(kind="bar")  # Add a border to bars

plt.title("Distribution of Sports", fontsize=15)  # More prominent title
plt.xlabel("Sports", fontsize=12)  # X-axis label
plt.ylabel("Number of Participants", fontsize=12)  # Y-axis label
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
plt.tight_layout()  # Adjust layout to prevent cutting off labels

#plt.show()
plt.savefig("./Plots/Distribution_Of_Sports.png", dpi=300)  # Higher resolution

sport_counts = sports.value_counts()

plt.clf()
# Plotting the Pie Chart
sport_counts.plot(kind="pie")  # Add a border to bars

plt.title("Distribution of Sports", fontsize=15)  # More prominent title
plt.xlabel("Sports", fontsize=12)  # X-axis label
#plt.ylabel("Number of Participants", fontsize=12)  # Y-axis label
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
plt.tight_layout()  # Adjust layout to prevent cutting off labels

#plt.show()
plt.savefig("./Plots/Distribution_Of_Sports_pie.png", dpi=300)  # Higher resolution
