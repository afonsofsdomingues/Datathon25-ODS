import nest_asyncio
import asyncio
from opendeepsearch import OpenDeepSearchTool
from opendeepsearch.prompts import REACT_PROMPT
from smolagents import LiteLLMModel, ToolCallingAgent, CodeAgent
import os
from dotenv import load_dotenv
import json
import pandas as pd

from typing import Optional

from litellm import completion

from opendeepsearch import OpenDeepSearchTool
# from opendeepsearch.wolfram_tool import WolframAlphaTool
from opendeepsearch.prompts import REACT_PROMPT, CONSTRAINTS_PROMPT, CONSTRAINTS_SATISFIED_PROMPT, CRITIQUE_PROMPT, \
    FEEDBACK_PROMPT, SEARCH_SYSTEM_PROMPT
from smolagents import LiteLLMModel, ToolCallingAgent, Tool, CodeAgent
import os
from dotenv import load_dotenv
import json
import pandas as pd

load_dotenv(override=True)
nest_asyncio.apply()  # Enable nested event loops

def create_codeAgent():
    model_name = "fireworks_ai/llama-v3p1-70b-instruct"
    search_agent = OpenDeepSearchTool(
        model_name=model_name,
        reranker="jina"
    )

    model = LiteLLMModel(
        model_name,
        temperature=0.2
    )

    code_agent = CustomCodeAgent(tools=[search_agent], model=model)
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
        result = asyncio.to_thread(agent.run, query)
        return [query, result]

    # Create tasks for each query and run them concurrently
    tasks = [await run_query(query) for query in queries]
    results = await asyncio.gather(*tasks)
    df = pd.DataFrame(results, columns=["original_question", "answer"])
    return df

async def run_codeAgent_async(queries: list[str]) -> pd.DataFrame:
    model_name = "fireworks_ai/llama-v3p1-70b-instruct"

    async def run_query(query: str) -> list:
        # Create a fresh agent for each query to avoid state leakage.
        agent = create_codeAgent()

        # Decompose the instruction into constraints
        constraints = await decompose_instruction(model_name, query)
        print(f"Constraints: {constraints}")

        feedback = None
        max_refinements = 3
        counter = 0

        def build_messages(context: str, query: str, feedback: Optional[dict] = None):
            messages = [{"role": "system", "content": SEARCH_SYSTEM_PROMPT}]
            if feedback:
                messages.append({
                    "role": "user",
                    "content": FEEDBACK_PROMPT.format(instruction=query,
                                                      previous_response=feedback["previous_response"],
                                                      feedback=feedback["feedback"])
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                })
            return messages

        result = asyncio.to_thread(agent.run, query)

        answer = result["answer"]
        context = result["contexts"]

        # print(f"Answer: {answer}")
        # print(f"Context: {context}")

        for refinement in range(max_refinements + 1):

            messages = build_messages(context, query, feedback)
            response = await completion(
                model=model_name,
                messages=messages,
                temperature=0.2
            )
            answer = response.choices[0].message.content

            # --- CRITIQUE STAGE ---
            if await constraints_satisfied(model_name, answer, constraints):
                return [query, answer]  # ✅ Successful answer

            # --- REFINE STAGE ---
            feedback = await critique_constraints(model_name, answer, constraints)
            # print(f"Feedback : {feedback}")

            # If still not satisfied after all refinements, return last version
            counter += 1
            # print(counter)
        return [query, answer]

        # result = await asyncio.to_thread(agent.run, query)
        # return [query, result]

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

async def decompose_instruction(model, query: str) -> list[str]:
    """Break down the instruction into constraints."""
    # Prepare messages for the LLM
    messages = [
        {"role": "user", "content": CONSTRAINTS_PROMPT.format(instruction=query)},
    ]
    response = completion(
        model=model,
        messages=messages,
        temperature=0.1  # Lower temp for more deterministic decomposition
    )

    # Parse the response into a list of constraints
    constraints = [c.strip() for c in response.choices[0].message.content.split('\n') if c.strip()]
    return constraints

async def constraints_satisfied(model, answer: str, constraints: list[str]) -> bool:
    prompt = CONSTRAINTS_SATISFIED_PROMPT.format(
        constraints='\n'.join(f"- {c}" for c in constraints),
        answer=answer
    )
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return "yes" in response.choices[0].message.content.strip().lower()

async def critique_constraints(model, answer: str, constraints: list[str]) -> dict:
    prompt = CRITIQUE_PROMPT.format(
        constraints='\n'.join(f"- {c}" for c in constraints),
        answer=answer
    )
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return {
        "previous_response": answer,
        "feedback": response.choices[0].message.content.strip()
    }

class CustomCodeAgent(CodeAgent):
    def __init__(self, tools, model):
        # Initialize the parent class with the necessary parameters
        super().__init__(tools=tools, model=model)
        self.opendeepsearchtool = tools[0]
        # print("opendeepsearchtool", self.opendeepsearchtool)

    def run(self, query: str):
        # Call the original run method
        answer = super().run(query)
        # print(f"Answer: {answer}")

        contexts = self.opendeepsearchtool.get_context()
        # print(f"Contexts: {contexts}")

        # You can return a custom dictionary or structure
        return {
            'answer': answer,
            'contexts': contexts
        }

async def main():
    df_benchmark = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t", index_col=0)
    prompts = df_benchmark["Prompt"].tolist()
    # For demonstration, we only run the first prompt. You can extend this as needed.
    result_df = await run_codeAgent_async(prompts[:4])
    processed_df = merge_df_with_true_answers(result_df, df_benchmark)
    save_df_to_json(processed_df, "results.jsonl")
    print(processed_df)

if __name__ == "__main__":
    asyncio.run(main())
