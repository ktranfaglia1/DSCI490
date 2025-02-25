from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import os
import json
import numpy as np
import pandas as pd


def parse_concussions(dataInput):

    if type(dataInput) != str:
        return np.nan

    OPEN_API_KEY = os.environ["OPEN_AI_KEY"]

    system_Message = """
        You are a data parsing agent responsible for formatting unstructured sports player data into structured JSON.

        ### Instructions:
        - Extract relevant details such as **Sport, Number of concussions**.
        - If the **Sport** is missing, use `"NaN"`.

        ### Example Input:
        Soccer: 2
        Football: 1

        ### Expected Output:
        [{{"Sport": "Soccer", "Concussions": 2}},
        {{"Sport": "Football", "Concussions": 1}}]
    """.strip()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPEN_API_KEY,
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_Message), ("human", "{dataInput}")]
    )

    parser = StrOutputParser()

    chain = prompt_template | llm | parser

    return chain.invoke({"dataInput": dataInput})


print(parse_concussions("yes - Taekwondo/MMA/Kickboxing 3"))
df_filtered = pd.read_csv("./Data/FilteredData.csv")
df_filtered["Q78_1_TEXT"] = df_filtered["Q78_1_TEXT"].apply(parse_concussions)
df_filtered.to_csv("./Data/FilteredData2.csv", index=False)
