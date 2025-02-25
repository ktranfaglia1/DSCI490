import matplotlib.pyplot as plt
import pandas as pd
import json
import regex as re

initial_data = pd.read_csv("../Data/CompleteData.csv")




def addSports(dataframe):
    sports = []
    for sportJson in dataframe:
        if re.search(r"\[.*\]", sportJson):
            for sport in json.loads(sportJson):
                sports.append(sport["Sport"])
    return sports

sports = pd.Series(addSports(initial_data["Q77"]))
sports = sports.str.strip().str.lower()

def set_regex(regex, set_value, element):
    if re.search(regex, element):
        return set_value
    else:
        return element


sports = sports.apply(lambda x: set_regex(".*cross.*country.*", "cross country", x))
sports = sports.apply(lambda x: set_regex(".*hockey.*", "hockey", x))
sports = sports.apply(lambda x: set_regex(".*track.*", "track", x))
sports = sports.apply(lambda x: set_regex(".*cheer.*", "cheer", x))
sports = sports.apply(lambda x: set_regex(".*footbal.*", "football", x))
sports = sports.apply(lambda x: set_regex(".*swim.*", "swim", x))

sports = sports[sports != 'nan']


# Get the value counts for the pie chart
sport_counts = sports.value_counts()

# Identify sports with counts less than 3
small_sports = sport_counts[sport_counts < 3].index

# Merge those sports into 'Other'
sports = sports.apply(lambda x: "Other" if x in small_sports else x)

sport_counts = sports.value_counts()

# Plotting the Pie Chart
plt.figure(figsize=(8, 8))  # Set the figure size
plt.pie(sport_counts, labels=sport_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)

# Make the chart look better
plt.title('Distribution of Sports')  # Add a title
plt.axis('equal')  # Ensure the pie chart is a circle

# Show the chart
#plt.show()
plt.savefig("../graphs/Distibution_Of_sports_Being_Played")