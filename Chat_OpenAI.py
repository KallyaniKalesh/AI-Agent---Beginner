"""
AI Data Analysis Agent
LangChain + Azure OpenAI
Author: Kallyani Kalesh
"""

import os
import urllib.request
import pandas as pd
from IPython.display import Markdown, display

from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType



# Model Setup
llm = AzureChatOpenAI(
    azure_deployment="gpt-4-1106",
    openai_api_version="2024-04-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0
)



# Model Sanity Check
test_message = HumanMessage(
    content=(
        "Translate this sentence from English to French and Spanish: "
        "I like red cars and blue houses, but my dog is yellow."
    )
)

print(llm.invoke([test_message]).content)



# Data Loading


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

CSV_URL = "https://covidtracking.com/data/download/all-states-history.csv"
CSV_PATH = os.path.join(DATA_DIR, "all-states-history.csv")

urllib.request.urlretrieve(CSV_URL, CSV_PATH)

df = pd.read_csv(CSV_PATH).fillna(0)



# Agent Creation

agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)



# Analysis Prompt
                                                                                                                                                           

CSV_PROMPT_PREFIX = """
Inspect the DataFrame and identify relevant columns.
"""

CSV_PROMPT_SUFFIX = """
Rules:
- Use at least two calculation methods
- Compare results for consistency
- Retry if inconsistent
- Use only computed values
- Present results in Markdown

Final Answer must include:

Explanation:
Describe the steps and columns used.
"""

QUESTION = (
    "How many patients were hospitalized during July 2020 "
    "in Texas, and what was the nationwide total across all states? "
    "Use the hospitalizedIncrease column."
)



# Agent Execution

response = agent.invoke(
    CSV_PROMPT_PREFIX + QUESTION + CSV_PROMPT_SUFFIX
)

display(Markdown(response["output"]))
