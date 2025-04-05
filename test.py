from opendeepsearch import OpenDeepSearchTool
import os
from dotenv import load_dotenv

load_dotenv()

# Using Serper (default)
search_agent = OpenDeepSearchTool(
    model_name="fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct",
    reranker="jina"
)

# Or using SearXNG
# search_agent = OpenDeepSearchTool(
#     model_name="openrouter/google/gemini-2.0-flash-001",
#     reranker="jina",
#     search_provider="searxng",
#     searxng_instance_url="https://your-searxng-instance.com",
#     searxng_api_key="your-api-key-here"  # Optional
# )

if not search_agent.is_initialized:
    search_agent.setup()
    
query = "Fastest land animal?"
result = search_agent.forward(query)
print(result)