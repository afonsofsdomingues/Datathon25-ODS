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
    model_name = "fireworks_ai/llama-v3p1-70b-instruct"
    search_agent = OpenDeepSearchTool(
        model_name=model_name,
        reranker="jina"
    )

    model = LiteLLMModel(
        model_name,
        temperature=0.2
    )

    # code_agent = CodeAgent(tools=[search_agent], model=model)
    # result = code_agent.run(query)

    code_agent = CustomCodeAgent(tools=[search_agent], model=model)

    results = []

    for query in queries:

        search_agent.flush_contexts()
        code_agent.opendeepsearchtool.flush_contexts()

        # Decompose the instruction into constraints
        constraints = decompose_instruction(model_name, query)
        print(f"Constraints: {constraints}")

        feedback = None
        max_refinements = 2

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

        result = code_agent.run(query)

        answer = result["answer"]
        context = result["contexts"]

        # print(f"Answer: {answer}")
        # print(f"Context: {context}")

        for refinement in range(max_refinements+1):

            messages = build_messages(context, query, feedback)
            response = completion(
                model=model_name,
                messages=messages,
                temperature=0.2
            )
            answer = response.choices[0].message.content

            # --- CRITIQUE STAGE ---
            if constraints_satisfied(model_name, answer, constraints):
                results.append([query, answer])
                break # ✅ Successful answer

            # --- REFINE STAGE ---
            feedback = critique_constraints(model_name, answer, constraints)
            # print(f"Feedback : {feedback}")

        results.append([query, answer])

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

def decompose_instruction(model, query: str) -> list[str]:
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

def constraints_satisfied(model, answer: str, constraints: list[str]) -> bool:
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

def critique_constraints(model, answer: str, constraints: list[str]) -> dict:
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

#Add other agents if you want....
# def new_agent(queries: list[str]):
#     pass

def save_df_to_json(df, filename):
    df.to_json(filename, orient="records", lines=True)


df_benchmark = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t", index_col=0)
prompts = df_benchmark["Prompt"].tolist()


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

if __name__ == "__main__":
    # Example usage
    result_df = run_codeAgent(prompts) # run_react(prompts)

    # processed_df = merge_df_with_true_answers(result_df, df_benchmark)
    # save_df_to_json(processed_df, "results.jsonl")


