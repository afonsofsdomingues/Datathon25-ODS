import pandas as pd
import json

df_results = pd.read_json("results.jsonl", orient='records', lines=True)

print("df_results:")
print(df_results)

correct = df_results[df_results["final_grade"].isin(["A"])]

#num of rows in correct
num_correct = len(correct)

print("pseudo accuracy: ", num_correct / (len(df_results)))