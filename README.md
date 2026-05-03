# 🔍 OpenDeepSearch (Datathon 2025 Fork)

<div align="center">
  <img src="./assets/sentient-logo-narrow.png" alt="OpenDeepSearch Logo" width="55%"/>
</div>

### Datathon 2025 — Team *Siuuupremacy* — **Rank:** Top 4 🏆 

**Team Members:** Afonso Domingues, Suei-Wen Chen, Thösam Norlha-Tsang and Karlo Angelic

> **️⚠️ Note**: This repository is a customized fork of **[OpenDeepSearch](https://github.com/sentient-agi/OpenDeepSearch)** created specifically for the **Datathon 2025** competition.

## Project Resources 📁
- [📄 Project Report](./Datathon_Report.pdf)  
- [🎤 Presentation Slides](./Datathon_Presentation.pdf)

## Our Approach (Datathon 2025) 🧠

Our team focused on improving OpenDeepSearch's performance on the **FRAMES dataset**, specifically targeting:  
- **Multiple-constraint questions**  
- **Numerical reasoning**  

We explored two main approaches during the hackathon:  

1. **DeCRIM Pipeline** (Decompose → Critique → Refine): Enables self-correction & constraint validation.  
2. **Enhanced Query Breakdown**: Refined instructions for stronger multi-hop reasoning.  

*Due to API limitations, we couldn’t fully integrate both approaches. However, using our **query breakdown + custom reranker** strategies, we achieved a highly competitive score of **0.6529**.*

---

## About OpenDeepSearch 📝

OpenDeepSearch (ODS) is a lightweight yet powerful search tool designed for seamless integration with AI agents. It enables deep web search and retrieval, optimized for use with Hugging Face's **[SmolAgents](https://github.com/huggingface/smolagents)** ecosystem.

### Key Features ✨
- **Semantic Search** 🧠: Leverages **Crawl4AI** and semantic search rerankers (like Jina AI) to provide in-depth results.
- **Two Search Modes** ⚡: **Default Mode** for quick single-hop queries, and **Pro Mode** for deep search and complex multi-hop queries.
- **Optimized for AI Agents** 🤖: Works seamlessly with SmolAgents like `CodeAgent`.

## Quick Start 🚀

### 1. Installation

```bash
pip install -e .
pip install -r requirements.txt
```

### 2. Setup API Keys

You will need keys for Serper (search), Jina (reranker), and your LLM provider (e.g., OpenRouter).

```bash
export SERPER_API_KEY='your-serper-api-key'
export JINA_API_KEY='your-jina-api-key'
export OPENROUTER_API_KEY='your-openrouter-key'
```

### 3. Usage Example

```python
from opendeepsearch import OpenDeepSearchTool
import os

search_agent = OpenDeepSearchTool(
    model_name="openrouter/google/gemini-2.0-flash-001",
    reranker="jina"
)

if not search_agent.is_initialized:
    search_agent.setup()
    
result = search_agent.forward("Fastest land animal?")
print(result)
```

## Acknowledgments 💡

This fork builds upon the original **OpenDeepSearch** project by [Sentient Foundation](https://github.com/sentient-agi). 
- 📄 Original Paper: [Open Deep Search: Democratizing Search with Open-source Reasoning Agents](https://arxiv.org/pdf/2503.20201)
- 🤗 Built using [SmolAgents](https://huggingface.co/docs/smolagents/index), [Crawl4AI](https://github.com/unclecode/crawl4ai), and [LiteLLM](https://www.litellm.ai/).
