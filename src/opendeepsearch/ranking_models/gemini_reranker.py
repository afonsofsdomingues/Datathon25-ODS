from abc import ABC, abstractmethod
import torch
from typing import List, Dict, Union
from litellm import embedding
import torch
import requests
import json
from typing import List
from opendeepsearch.ranking_models.base_reranker import BaseSemanticSearcher
import openai
import os

class GeminiSemanticSearcher(BaseSemanticSearcher):
    """
    Gemini class for semantic search implementations.
    
    This class defines the interface that all semantic searchers must implement.
    Subclasses should implement the _get_embeddings method according to their
    specific embedding source.
    """

    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            torch.Tensor containing the embeddings shape: (num_texts, embedding_dim)
        """

        # print("Calculating embeddings...")
        client = openai.OpenAI(
            base_url = "https://api.fireworks.ai/inference/v1",
            api_key=os.getenv("FIREWORKS_API_KEY"),
        )
        # print("Client: ", client)
        response = client.embeddings.create(
            model="nomic-ai/nomic-embed-text-v1.5",
            input=texts,
            dimensions=1024
        )
        # print("Response: ", response)
        # print("Response data: ", response.data)
        embeddings_list = [item.embedding for item in response.data]
        embeddings = torch.tensor(embeddings_list)
        return embeddings

        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = embedding(model="gemini/text-embedding-004", input=batch)
            
            # Extract embeddings from response
            batch_embeddings = [item['embedding'] for item in response['data']]
            all_embeddings.extend(batch_embeddings)

        # Convert to torch tensor
        embeddings = torch.tensor(all_embeddings)
        
        return embeddings