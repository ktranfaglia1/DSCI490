from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import os
import json
import numpy as np


def parse_sports(dataInput):

    if type(dataInput) != str:
        return np.nan

    OPEN_API_KEY = os.environ["OPEN_AI_KEY"]

    system_Message = """
        You are a data parsing agent responsible for formatting unstructured sports player data into structured JSON.

        ### Instructions:
        - Extract relevant details such as **Sport, Position, and Years Played**.
        - If the **position** is missing, use `"NaN"`.

        ### Example Input:
        Football, Running Back, all 4 years. Baseball, outfield, 3 years. Track, all 4 years.

        ### Expected Output:
        [{{"Sport": "Football", "Years": 4, "Position": "Running Back"}},
        {{"Sport": "Baseball", "Years": 3, "Position": "Outfield"}},
        {{"Sport": "Track", "Years": 4, "Position": "NaN"}}]
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


print(parse_sports("Soccer, Tennis, all 4 years"))
