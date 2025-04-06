import nest_asyncio
import asyncio
from opendeepsearch import OpenDeepSearchTool
from opendeepsearch.prompts import REACT_PROMPT
from smolagents import LiteLLMModel, ToolCallingAgent, CodeAgent
import os
from dotenv import load_dotenv
import json
import pandas as pd
from opendeepsearch.prompts import SEARCH_SYSTEM_PROMPT, CONSTRAINTS_PROMPT, CONSTRAINTS_SATISFIED_PROMPT, CRITIQUE_PROMPT, FEEDBACK_PROMPT

from litellm import completion, utils
from opendeepsearch.prompts import SEARCH_SYSTEM_PROMPT, CONSTRAINTS_PROMPT, CONSTRAINTS_SATISFIED_PROMPT, CRITIQUE_PROMPT, FEEDBACK_PROMPT, CONSTRAINTS_SATISFIED_META_PROMPT


load_dotenv(override=True)

os.getenv("FIREWORKS_API_KEY", "fw_3ZGfS7pC4qA4kiTZ3w23qXvD")
nest_asyncio.apply()  # Enable nested event loops

def create_codeAgent(constraints: list[str] = None):
    search_agent = OpenDeepSearchTool(
        model_name="fireworks_ai/llama-v3p1-70b-instruct",
        # reranker="jina",
        constraints=constraints,
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
        reranker="jina"
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
        agent = create_react_agent()
        result = await asyncio.to_thread(agent.run, query)
        return [query, result]

    # Create tasks for each query and run them concurrently
    tasks = [run_query(query) for query in queries]
    results = await asyncio.gather(*tasks)
    df = pd.DataFrame(results, columns=["original_question", "answer"])
    return df

async def run_codeAgent_async(queries: list[str], max_concurrent_tasks: int = 5) -> pd.DataFrame:
    sem = asyncio.Semaphore(max_concurrent_tasks)

    async def run_query(query: str) -> list:
        async with sem:
            # Create a fresh agent for each query to avoid state leakage.
            constraints = await decompose_instruction(query)
            agent = create_codeAgent(constraints)
            result = await asyncio.to_thread(agent.run, query)
            return [query, result]

    # Create tasks for each query and run them concurrently with concurrency limits.
    tasks = [run_query(query) for query in queries]
    results = await asyncio.gather(*tasks)
    df = pd.DataFrame(results, columns=["original_question", "answer"])
    return df

async def decompose_instruction(query: str) -> list[str]:
    """Break down the instruction into constraints."""
    # Prepare messages for the LLM
    messages = [
        {"role": "user", "content": CONSTRAINTS_PROMPT.format(instruction=query)},
    ]
    print("CONSTRAINTS PROMPT: ", CONSTRAINTS_PROMPT.format(instruction=query))
    response = completion(
        model="fireworks_ai/llama-v3p1-70b-instruct",
        messages=messages,
        temperature=0.1  # Lower temp for more deterministic decomposition
    )

    # Parse the response into a list of constraints
    constraints = [c.strip() for c in response.choices[0].message.content.split('\n') if c.strip()]
    print(f"[CONSTRAINTS]: {constraints}")
    return constraints

def merge_df_with_true_answers(df, df_benchmark):
    df_merged = pd.merge(df, df_benchmark, left_on="original_question", right_on="Prompt", how="inner")
    df_essential = df_merged[['original_question', 'answer', 'Answer']]
    df_renamed = df_essential.rename(columns={"Answer": "true_answer"})
    return df_renamed

def save_df_to_json(df, filename):
    df.to_json(filename, orient="records", lines=True)

async def main():
    df_benchmark = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t", index_col=0)
    prompts = df_benchmark["Prompt"].tolist()[400:500]
    # Adjust max_concurrent_tasks to limit the number of parallel tasks.
    result_df = await run_codeAgent_async(prompts, max_concurrent_tasks=4)
    processed_df = merge_df_with_true_answers(result_df, df_benchmark)
    save_df_to_json(processed_df, "results.jsonl")
    print(processed_df)

if __name__ == "__main__":
    asyncio.run(main())