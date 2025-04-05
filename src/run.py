from opendeepsearch import OpenDeepSearchTool
# from opendeepsearch.wolfram_tool import WolframAlphaTool
from opendeepsearch.prompts import REACT_PROMPT
from smolagents import LiteLLMModel, ToolCallingAgent, Tool
import os
from dotenv import load_dotenv
import json
import pandas as pd

load_dotenv(override=True)



def run(queries: list[str]):
    """
    Run the OpenDeepSearch agent with the provided query.

    Args:
        queries (list[str]): The queries to send to the agent.

    Returns:
        str: The response from the agent.
    """
    # Initialize the LiteLLM model
    model = LiteLLMModel(
        "fireworks_ai/llama-v3p1-70b-instruct",  # Your Fireworks Deepseek model
        temperature=0.7
    )
    search_agent = OpenDeepSearchTool(model_name="fireworks_ai/llama-v3p1-70b-instruct", reranker="jina") # Set reranker to "jina" or "infinity"

    # Initialize the Wolfram Alpha tool
    # wolfram_tool = WolframAlphaTool(app_id=os.environ["WOLFRAM_ALPHA_APP_ID"])

    # Initialize the React Agent with search and wolfram tools
    react_agent = ToolCallingAgent(
        tools=[search_agent],
        model=model,
        prompt_templates=REACT_PROMPT # Using REACT_PROMPT as system prompt
    )
    results = []
    for query in queries:
        result = react_agent.run(query)
        results.append([query, result])

    #Create dataframe with 2 columns: query and result
    df = pd.DataFrame(results, columns=["query", "result"])

    return df

def save_df_to_json(df, filename):
    df.to_json(filename, orient="records", lines=True)

if __name__ == "__main__":
    # Example usage
    query = ["What is the distance, in metres, between the Colosseum in Rome and the Rialto bridge in Venice"]
    result = run(query)
    save_df_to_json(result, "results.json")