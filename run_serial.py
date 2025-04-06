from opendeepsearch import OpenDeepSearchTool
# from opendeepsearch.wolfram_tool import WolframAlphaTool
from opendeepsearch.prompts import REACT_PROMPT
from smolagents import LiteLLMModel, ToolCallingAgent, Tool, CodeAgent
import os
from dotenv import load_dotenv
import json
import pandas as pd

load_dotenv(override=True)



def run_react(queries: list[str]):
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
    search_agent = OpenDeepSearchTool(
        model_name="fireworks_ai/llama-v3p1-70b-instruct", 
        reranker="jina"
    ) # Set reranker to "jina" or "infinity"

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
    df = pd.DataFrame(results, columns=["original_question", "answer"])

    return df

def run_codeAgent(queries: list[str]):
    search_agent = OpenDeepSearchTool(
        model_name="fireworks_ai/llama-v3p1-70b-instruct",
        reranker="jina"
    )

    model = LiteLLMModel(
        "fireworks_ai/llama-v3p1-70b-instruct",
        temperature=0.2
    )

    code_agent = CodeAgent(tools=[search_agent], model=model)
    results = []
    for query in queries:
        result = code_agent.run(query)
        results.append([query, result])

    df = pd.DataFrame(results, columns=["original_question", "answer"])

    return df

def merge_df_with_true_answers(df, df_benchmark):
    print("my df:")
    print(df)
    print()
    print("benchmark df: ")
    print(df_benchmark)
    #Merge the two df:s based on left: original_question, right: Prompt
    df_merged = pd.merge(df, df_benchmark, left_on="original_question", right_on="Prompt", how="inner")
    #Keep original_question, answer, Answer
    df_essential = df_merged[['original_question','answer', 'Answer']] #answer is our results, Answer is Ground truth from benchmark
    df_renamed = df_essential.rename(columns={"Answer": "true_answer"})
    return df_renamed

#Add other agents if you want....
# def new_agent(queries: list[str]):
#     pass

def save_df_to_json(df, filename):
    df.to_json(filename, orient="records", lines=True)


df_benchmark = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t", index_col=0)
prompts = df_benchmark["Prompt"].tolist()[:4]

if __name__ == "__main__":
    # Example usage
    result_df = run_react(prompts)
    processed_df = merge_df_with_true_answers(result_df, df_benchmark)
    save_df_to_json(processed_df, "results.jsonl")