# from opendeepsearch import OpenDeepSearchTool
# # from opendeepsearch.wolfram_tool import WolframAlphaTool
# from opendeepsearch.prompts import REACT_PROMPT
# from smolagents import LiteLLMModel, ToolCallingAgent, Tool, CodeAgent
# import os
# from dotenv import load_dotenv
# import json
# import pandas as pd
# from litellm import completion, utils
# from opendeepsearch.prompts import SEARCH_SYSTEM_PROMPT, CONSTRAINTS_PROMPT, CONSTRAINTS_SATISFIED_PROMPT, CRITIQUE_PROMPT, FEEDBACK_PROMPT, CONSTRAINTS_SATISFIED_META_PROMPT

# load_dotenv(override=True)



# def run_react(queries: list[str]):
#     """
#     Run the OpenDeepSearch agent with the provided query.

#     Args:
#         queries (list[str]): The queries to send to the agent.

#     Returns:
#         str: The response from the agent.
#     """
#     # Initialize the LiteLLM model
#     model = LiteLLMModel(
#         "fireworks_ai/llama-v3p1-70b-instruct",  # Your Fireworks Deepseek model
#         temperature=0.7
#     )
#     search_agent = OpenDeepSearchTool(
#         model_name="fireworks_ai/llama-v3p1-70b-instruct", 
#         #reranker="jina"
#     ) # Set reranker to "jina" or "infinity"

#     # Initialize the Wolfram Alpha tool
#     # wolfram_tool = WolframAlphaTool(app_id=os.environ["WOLFRAM_ALPHA_APP_ID"])

#     # Initialize the React Agent with search and wolfram tools
#     react_agent = ToolCallingAgent(
#         tools=[search_agent],
#         model=model,
#         prompt_templates=REACT_PROMPT # Using REACT_PROMPT as system prompt
#     )
#     results = []
#     for query in queries:
#         result = react_agent.run(query)
#         results.append([query, result])

#     #Create dataframe with 2 columns: query and result
#     df = pd.DataFrame(results, columns=["original_question", "answer"])

#     return df

# def decompose_instruction(query: str) -> list[str]:
#     """Break down the instruction into constraints."""
#     # Prepare messages for the LLM
#     messages = [
#         {"role": "user", "content": CONSTRAINTS_PROMPT.format(instruction=query)},
#     ]
#     print("CONSTRAINTS PROMPT: ", CONSTRAINTS_PROMPT.format(instruction=query))
#     response = completion(
#         model="fireworks_ai/llama-v3p1-70b-instruct",
#         messages=messages,
#         temperature=0.1  # Lower temp for more deterministic decomposition
#     )

#     # Parse the response into a list of constraints
#     constraints = [c.strip() for c in response.choices[0].message.content.split('\n') if c.strip()]
#     return constraints

# def run_codeAgent(queries: list[str]):
#     search_agent = OpenDeepSearchTool(
#         model_name="fireworks_ai/llama-v3p1-70b-instruct",
#         #reranker="jina"
#     )

#     model = LiteLLMModel(
#         "fireworks_ai/llama-v3p1-70b-instruct",
#         temperature=0.2
#     )


#     code_agent = CodeAgent(tools=[search_agent], model=model)
#     results = []
#     for query in queries:

#         constraints = decompose_instruction(query)
#         print(f"[CONSTRAINTS]: {constraints}")

#         result = code_agent.run(query)

#         print(f"[RESULT]: {result}")

#         # print(f"[CONSTRAINTS SATISFIED PROMPT]: {CONSTRAINTS_SATISFIED_PROMPT.format(constraints=constraints, answer=result)}")
#         satisfied_query = CONSTRAINTS_SATISFIED_META_PROMPT.format(constraints=constraints, answer=result)
#         print(f"[SATISFIED QUERY]: {satisfied_query}")
#         satisfied_result = code_agent.run(satisfied_query)
#         print(f"[SATISFIED RESULT]: {satisfied_result}")
#         exit()


#         results.append([query, result])

#     df = pd.DataFrame(results, columns=["original_question", "answer"])

#     return df

# def merge_df_with_true_answers(df, df_benchmark):
#     print("my df:")
#     print(df)
#     print()
#     print("benchmark df: ")
#     print(df_benchmark)
#     #Merge the two df:s based on left: original_question, right: Prompt
#     df_merged = pd.merge(df, df_benchmark, left_on="original_question", right_on="Prompt", how="inner")
#     #Keep original_question, answer, Answer
#     df_essential = df_merged[['original_question','answer', 'Answer']] #answer is our results, Answer is Ground truth from benchmark
#     df_renamed = df_essential.rename(columns={"Answer": "true_answer"})
#     return df_renamed

# #Add other agents if you want....
# # def new_agent(queries: list[str]):
# #     pass

# def save_df_to_json(df, filename):
#     df.to_json(filename, orient="records", lines=True)


# df_benchmark = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t", index_col=0)
# prompts = df_benchmark["Prompt"].tolist()[:2]

# if __name__ == "__main__":
#     # Example usage
#     result_df = run_codeAgent(prompts)
#     processed_df = merge_df_with_true_answers(result_df, df_benchmark)
#     save_df_to_json(processed_df, "results.jsonl")

from opendeepsearch import OpenDeepSearchTool
# from opendeepsearch.wolfram_tool import WolframAlphaTool
from opendeepsearch.prompts import REACT_PROMPT
from smolagents import LiteLLMModel, ToolCallingAgent, Tool, CodeAgent
import os
from dotenv import load_dotenv
import json
import pandas as pd
from litellm import completion, utils
from opendeepsearch.prompts import SEARCH_SYSTEM_PROMPT, CONSTRAINTS_PROMPT, CONSTRAINTS_SATISFIED_PROMPT, CRITIQUE_PROMPT, FEEDBACK_PROMPT, CONSTRAINTS_SATISFIED_META_PROMPT

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
        #reranker="jina"
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

def decompose_instruction(query: str) -> list[str]:
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
    return constraints

num_retries = 2

def run_codeAgent(queries: list[str]):
    search_agent = OpenDeepSearchTool(
        model_name="fireworks_ai/llama-v3p1-70b-instruct",
        #reranker="jina"
    )



    
    results = []
    for query in queries:
        result = ""
        constraints = decompose_instruction(query)
        print(f"[CONSTRAINTS]: {constraints}")
        i = 0
        current_temperature = 0.2
        model = LiteLLMModel(
            "fireworks_ai/llama-v3p1-70b-instruct",
            temperature=current_temperature
        )
        code_agent = CodeAgent(tools=[search_agent], model=model)
        while result == "" and i < num_retries:
            i += 1

            tmp_res = code_agent.run(query)

            print(f"[RESULT]: {tmp_res}")

            # print(f"[CONSTRAINTS SATISFIED PROMPT]: {CONSTRAINTS_SATISFIED_PROMPT.format(constraints=constraints, answer=result)}")
            satisfied_query = CONSTRAINTS_SATISFIED_META_PROMPT.format(instruction=query, constraints=constraints, answer=tmp_res)
            print(f"[SATISFIED QUERY]: {satisfied_query}")
            satisfied_result = code_agent.run(satisfied_query)
            print(f"[SATISFIED RESULT]: {satisfied_result}")
            if satisfied_result.lower() == "yes":
                print("Result was satisfied!")
                result = tmp_res
            elif i == num_retries:
                print("Max retries reached, moving to next query...")
                result = tmp_res
                break
            else:
                print("Result was not satisfied, trying again...")
                current_temperature += 0.1
                model = LiteLLMModel(
                    "fireworks_ai/llama-v3p1-70b-instruct",
                    temperature=current_temperature
                )
                code_agent = CodeAgent(tools=[search_agent], model=model)
                continue


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
prompts = df_benchmark["Prompt"].tolist()

if __name__ == "__main__":
    # Example usage
    result_df = run_codeAgent(prompts)
    processed_df = merge_df_with_true_answers(result_df, df_benchmark)
    save_df_to_json(processed_df, "results.jsonl")