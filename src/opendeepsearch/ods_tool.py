from typing import Optional, Literal
from smolagents import Tool
from opendeepsearch.ods_agent import OpenDeepSearchAgent

class OpenDeepSearchTool(Tool):
    name = "web_search"
    description = """
    Performs web search based on your query (think a Google search) then returns the final answer that is processed by an llm."""
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to perform",
        },
    }
    output_type = "string"

    def __init__(
        self,
        model_name: Optional[str] = None,
        reranker: str = "gemini",
        search_provider: Literal["serper", "searxng"] = "serper",
        serper_api_key: Optional[str] = None,
        searxng_instance_url: Optional[str] = None,
        searxng_api_key: Optional[str] = None
    ):
        super().__init__()
        self.search_model_name = model_name  # LiteLLM model name
        self.reranker = reranker
        self.search_provider = search_provider
        self.serper_api_key = serper_api_key
        self.searxng_instance_url = searxng_instance_url
        self.searxng_api_key = searxng_api_key

        self.contexts = ""

    def forward(self, query: str):
        self.setup()

        answer, context = self.search_tool.ask_sync(query, max_sources=2, pro_mode=True)

        self.contexts = self.contexts + "\n\n" + context
        return answer

    def setup(self):
        self.search_tool = OpenDeepSearchAgent(
            self.search_model_name,
            reranker=self.reranker,
            search_provider=self.search_provider,
            serper_api_key=self.serper_api_key,
            searxng_instance_url=self.searxng_instance_url,
            searxng_api_key=self.searxng_api_key
        )

    def flush_contexts(self):
        self.contexts = ""
        return True

    def get_context(self):
        return self.contexts
