import pandas as pd
import json
import typing
import re

originalDf = pd.read_csv("./Data/Labeled_survey_data.csv")
df = originalDf


def applyRegex(sport: str, searchValue: str, replaceValue: str) -> str:
    if re.match(searchValue, sport, re.IGNORECASE):
        return replaceValue
    return sport


def cleanSport(sport: str) -> str:

    sport = applyRegex(sport, ".*cross.*country.*", "cross country")
    sport = applyRegex(sport, "ice hockey", "hockey")
    sport = applyRegex(sport, ".*track.*", "track")
    sport = applyRegex(sport, ".*cheer.*", "cheer")
    sport = applyRegex(sport, ".*footbal.*", "football")
    sport = applyRegex(sport, ".*swim.*", "swim")

    return sport.lower()


players = originalDf["Sports_Info"]

for playerIndex, player in enumerate(players):
    try:
        sportJson = json.loads(player)
    except:
        continue

    for sportIndex, sport in enumerate(sportJson):
        sportJson[sportIndex]["Sport"] = cleanSport(sport["Sport"])
    players.iloc[playerIndex] = json.dumps(sportJson)

originalDf["Sports_Info"] = players
print(originalDf["Sports_Info"])

originalDf.to_csv("./Data/Labeled_survey_data.csv")
