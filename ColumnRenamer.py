#  Author: Kyle Tranfaglia
#  Title: DSCI490 - ColumnRenamer
#  Last updated:  02/24/25
#  Description: This program is a utility program to label all columns in a dataset with a semi-descriptive name
import pandas as pd

# load in dataset
df = pd.read_csv("./Data/The_survey_data.csv")

# Rename columns with descriptive labels
df.rename(columns={'Q64': 'Consent'}, inplace=True)
df.rename(columns={'Q65': 'Age_Confirmation'}, inplace=True)
df.rename(columns={'Q81': 'Age'}, inplace=True)
df.rename(columns={'Q71': 'DOB'}, inplace=True)
df.rename(columns={'Q1': 'Gender'}, inplace=True)
df.rename(columns={'Q72': 'Race'}, inplace=True)
df.rename(columns={'Q73': 'English_Speaker'}, inplace=True)
df.rename(columns={'Q74': 'Student_Status'}, inplace=True)
df.rename(columns={'Q75': 'Student_Year'}, inplace=True)
df.rename(columns={'Q86': 'Head_Injury_Status'}, inplace=True)
df.rename(columns={'Q87': 'Concussion_Status'}, inplace=True)
df.rename(columns={'Q88': 'Lose_Consciousness'}, inplace=True)
df.rename(columns={'Q76': 'Sports_Status'}, inplace=True)
df.rename(columns={'Q77': 'Sports_Info'}, inplace=True)
df.rename(columns={'Q78': 'Sports_Concussion_Status'}, inplace=True)
df.rename(columns={'Q78_1_TEXT': 'Sports_Concussion_Info'}, inplace=True)
df.rename(columns={'Q84': 'Non_Sports_Concussion_Status'}, inplace=True)
df.rename(columns={'Q84_1_TEXT': 'Non_Sports_Concussion_Info'}, inplace=True)

df.rename(columns={'Q2': 'Bed_Time'}, inplace=True)
df.rename(columns={'Q3': 'Min_To_Sleep'}, inplace=True)
df.rename(columns={'Q4': 'Wake_Up'}, inplace=True)
df.rename(columns={'Q5': 'Sleep_Per_Night'}, inplace=True)
df.rename(columns={'Q7': 'Cant_Sleep'}, inplace=True)
df.rename(columns={'Q8': 'Wake_In_Night'}, inplace=True)
df.rename(columns={'Q57': 'Wake_To_Bathroom'}, inplace=True)
df.rename(columns={'Q58': 'Breathe_Discomfort'}, inplace=True)
df.rename(columns={'Q59': 'Snore_Cough'}, inplace=True)
df.rename(columns={'Q60': 'Too_Cold'}, inplace=True)
df.rename(columns={'Q61': 'Too_Hot'}, inplace=True)
df.rename(columns={'Q62': 'Bad_Dreams'}, inplace=True)
df.rename(columns={'Q63': 'Pain'}, inplace=True)
df.rename(columns={'Q17': 'Sleep_Quality'}, inplace=True)
df.rename(columns={'Q18': 'Sleep_Meds'}, inplace=True)
df.rename(columns={'Q19': 'Staying_Awake_Issues'}, inplace=True)
df.rename(columns={'Q20': 'Motivation_Issues'}, inplace=True)
df.rename(columns={'Q21': 'Roomate'}, inplace=True)
df.rename(columns={'Q23': 'Loud_Snore'}, inplace=True)
df.rename(columns={'Q64.1': 'Breathe_Pause'}, inplace=True)
df.rename(columns={'Q65.1': 'Leg_Twitch'}, inplace=True)
df.rename(columns={'Q66': 'Sleep_Confusion'}, inplace=True)

df.rename(columns={'Q69_1': 'Noise_Concentration_Issues'}, inplace=True)
df.rename(columns={'Q69_2': 'Concentration_Issues'}, inplace=True)
df.rename(columns={'Q69_3': 'Surrounding_Concentration_Issues'}, inplace=True)
df.rename(columns={'Q69_4': 'Good_Music_Concentration'}, inplace=True)
df.rename(columns={'Q69_5': 'Concentration_Aware'}, inplace=True)
df.rename(columns={'Q69_6': 'Reading_Concentration_Issues'}, inplace=True)
df.rename(columns={'Q69_7': 'Trouble_Blocking_Thoughts'}, inplace=True)
df.rename(columns={'Q69_8': 'Excitement_Concentration_Issues'}, inplace=True)
df.rename(columns={'Q69_9': 'Ignore_Hunger_Concentrating'}, inplace=True)
df.rename(columns={'Q69_10': 'Good_Task_Switching'}, inplace=True)
df.rename(columns={'Q69_11': 'Long_Time_Focus'}, inplace=True)

df.rename(columns={'Q69_12': 'Poor_Listening_Writing'}, inplace=True)
df.rename(columns={'Q69_13': 'Quick_Interest'}, inplace=True)
df.rename(columns={'Q69_14': 'Easy_Read_Write_On_Phone'}, inplace=True)
df.rename(columns={'Q69_15': 'Trouble_Multiple_Conversations'}, inplace=True)
df.rename(columns={'Q69_16': 'Trouble_Quick_Creativity'}, inplace=True)
df.rename(columns={'Q69_17': 'Good_Interruption_Recovery'}, inplace=True)
df.rename(columns={'Q69_18': 'Good_Thought_Recovery'}, inplace=True)
df.rename(columns={'Q69_19': 'Good_Task_Alteration'}, inplace=True)
df.rename(columns={'Q69_20': 'Poor_Perspective_Thinking'}, inplace=True)

df.rename(columns={'Duration (in seconds)': 'Duration'}, inplace=True)
df.rename(columns={'Attention_score': 'Attention_Score'}, inplace=True)

# Save to a new file
df.to_csv("./Data/Labeled_survey_data.csv", index=False)
