import nest_asyncio
import asyncio
from opendeepsearch import OpenDeepSearchTool
from opendeepsearch.prompts import CODE_AGENT_PROMPT, REACT_PROMPT
from smolagents import LiteLLMModel, ToolCallingAgent, CodeAgent
import os
from dotenv import load_dotenv
import json
import pandas as pd

load_dotenv(override=True)
nest_asyncio.apply()  # Enable nested event loops

def create_codeAgent():
    search_agent = OpenDeepSearchTool(
        model_name="fireworks_ai/llama-v3p1-70b-instruct",
        # reranker="jina"
    )

    model = LiteLLMModel(
        "fireworks_ai/llama-v3p1-70b-instruct",
        temperature=0.2
    )

    code_agent = CodeAgent(tools=[search_agent], model=model)
    return code_agent


def create_react_agent():
    # Initialize the LiteLLM model
    model = LiteLLMModel(
        "fireworks_ai/llama-v3p1-70b-instruct",  # Your Fireworks Deepseek model
        temperature=0.7
    )
    search_agent = OpenDeepSearchTool(
        model_name="fireworks_ai/llama-v3p1-70b-instruct", 
        # reranker="jina"
    )
    # Initialize the React Agent with search tool
    react_agent = ToolCallingAgent(
        tools=[search_agent],
        model=model,
        prompt_templates=REACT_PROMPT
    )
    return react_agent

async def run_react_async(queries: list[str]) -> pd.DataFrame:

    async def run_query(query: str) -> list:
        # Create a fresh agent for each query to avoid state leakage.
        agent = create_codeAgent()
        result = await asyncio.to_thread(agent.run, query)
        return [query, result]

    # Create tasks for each query and run them concurrently
    tasks = [run_query(query) for query in queries]
    results = await asyncio.gather(*tasks)
    df = pd.DataFrame(results, columns=["original_question", "answer"])
    return df

async def run_codeAgent_async(queries: list[str]) -> pd.DataFrame:
    async def run_query(query: str) -> list:
        # Create a fresh agent for each query to avoid state leakage.
        agent = create_codeAgent()
        result = await asyncio.to_thread(agent.run, query)
        return [query, result]

    # Create tasks for each query and run them concurrently
    tasks = [run_query(query) for query in queries]
    results = await asyncio.gather(*tasks)
    df = pd.DataFrame(results, columns=["original_question", "answer"])
    return df

def merge_df_with_true_answers(df, df_benchmark):
    df_merged = pd.merge(df, df_benchmark, left_on="original_question", right_on="Prompt", how="inner")
    df_essential = df_merged[['original_question', 'answer', 'Answer']]
    df_renamed = df_essential.rename(columns={"Answer": "true_answer"})
    return df_renamed

def save_df_to_json(df, filename):
    df.to_json(filename, orient="records", lines=True)

async def main():
    df_benchmark = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t", index_col=0)
    prompts = df_benchmark["Prompt"].tolist()
    # For demonstration, we only run the first prompt. You can extend this as needed.
    result_df = await run_codeAgent_async(prompts)
    processed_df = merge_df_with_true_answers(result_df, df_benchmark)
    save_df_to_json(processed_df, "results.jsonl")
    print(processed_df)

if __name__ == "__main__":
    asyncio.run(main())
