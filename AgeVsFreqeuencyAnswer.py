import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('cleaned_SP_data.csv')

# Count the frequency of answers by age
counts = data.groupby(['Q81', 'Q17']).size().unstack()

# Plot the data as a stacked bar chart
counts.plot(kind='bar', stacked=True, figsize=(12, 6))

# Customize the plot
plt.title('Distribution of Sleep Quality Ratings by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title="Sleep Quality", bbox_to_anchor=(1,1))
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

'''
Instruction/idea:
- Q81 is the age column
- Q17 is an example of the survey questions
- Replace Q17 with any other survey question identifier you want to analyze
- The title should reflect the chosen question
- The graph shows the frequency of responses by age group
'''
