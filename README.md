# 🔍 OpenDeepSearch  
### Democratizing Search with Open-source Reasoning Models and Agents 🚀  

<div align="center">
  <img src="./assets/sentient-logo-narrow.png" alt="OpenDeepSearch Logo" width="55%"/>
</div>

<div align="center">
  <!-- Primary Links -->
  <a href="https://sentient.xyz/" target="_blank">
    <img alt="Homepage" src="https://img.shields.io/badge/Sentient-Homepage-%23EAEAEA?logo=google-chrome&logoColor=black"/>
  </a>
  <a href="https://github.com/sentient-agi" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/Github-sentient_agi-181717?logo=github"/>
  </a>
  <a href="https://huggingface.co/Sentientagi" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/🤗%20Hugging%20Face-SentientAGI-ffc107?color=ffc107&logoColor=white"/>
  </a>
</div>

<div align="center">
  <!-- Community Links -->
  <a href="https://discord.gg/sentientfoundation" target="_blank">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-SentientAGI-7289da?logo=discord&logoColor=white"/>
  </a>
  <a href="https://x.com/SentientAGI" target="_blank">
    <img alt="Twitter" src="https://img.shields.io/badge/X-SentientAGI-black?logo=x"/>
  </a>
</div>

<h4 align="center">
  📄 <a href="https://arxiv.org/pdf/2503.20201">Read the Paper</a>
</h4>

---

# Datathon 2025 — Team *Siuuupremacy* — **Rank:** Top 4 🏆 

## Table of Contents 📑
  - [Project Resources 📁](#project-resources-)
  - [Description 📝](#description-)
  - [Table of Contents 📑](#table-of-contents-)
  - [Features ✨](#features-)
  - [Approach Overview 🧠](#approach-overview-)
  - [Installation 📚](#installation-)
  - [Setup](#setup)
  - [Usage ️](#usage-️)
    - [Using OpenDeepSearch Standalone 🔍](#using-opendeepsearch-standalone-)
    - [Running the Gradio Demo 🖥️](#running-the-gradio-demo-️)
    - [Integrating with SmolAgents \& LiteLLM 🤖⚙️](#integrating-with-smolagents--litellm-️)
      - [](#)
    - [ReAct agent with math and search tools 🤖⚙️](#react-agent-with-math-and-search-tools-️)
      - [](#-1)
  - [Search Modes 🔄](#search-modes-)
    - [Default Mode ⚡](#default-mode-)
    - [Pro Mode 🔍](#pro-mode-)
  - [Acknowledgments 💡](#acknowledgments-)
  - [Citation](#citation)
  - [Contact 📩](#contact-)

---

## Project Resources 📁
- [📄 Project Report](./Datathon_Report.pdf)  
- [🎤 Presentation Slides](./Datathon_Presentation.pdf)

## Description 📝

OpenDeepSearch is a lightweight yet powerful search tool designed for seamless integration with AI agents. It enables deep web search and retrieval, optimized for use with Hugging Face's **[SmolAgents](https://github.com/huggingface/smolagents)** ecosystem.

<div align="center">
    <img src="./assets/evals.png" alt="Evaluation Results" width="80%"/>
</div>

- **Performance**: ODS performs on par with closed source search alternatives on single-hop queries such as [SimpleQA](https://openai.com/index/introducing-simpleqa/) 🔍.
- **Advanced Capabilities**: ODS performs much better than closed source search alternatives on multi-hop queries such as [FRAMES bench](https://huggingface.co/datasets/google/frames-benchmark) 🚀.

## Features ✨

- **Semantic Search** 🧠: Leverages **[Crawl4AI](https://github.com/unclecode/crawl4ai)** and semantic search rerankers (such as [Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct/tree/main) and [Jina AI](https://jina.ai/)) to provide in-depth results
- **Two Modes of Operation** ⚡:
  - **Default Mode**: Quick and efficient search with minimal latency.
  - **Pro Mode (Deep Search)**: More in-depth and accurate results at the cost of additional processing time.
- **Optimized for AI Agents** 🤖: Works seamlessly with **SmolAgents** like `CodeAgent`.
- **Fast and Lightweight** ⚡: Designed for speed and efficiency with minimal setup.
- **Extensible** 🔌: Easily configurable to work with different models and APIs.

## Approach Overview 🧠

Our team improved **OpenDeepSearch** performance on the **FRAMES dataset**, focusing on:  
- **Multiple-constraint questions**  
- **Numerical reasoning**  

We explored two main approaches:  

1. **DeCRIM Pipeline** (Decompose → Critique → Refine)  
   - Enables self-correction & constraint validation  
2. **Enhanced Query Breakdown**  
   - Refined instructions for stronger multi-hop reasoning  

⚠️ Due to API limitations, we couldn’t fully integrate both approaches.  
✅ However, our **query breakdown + custom reranker** achieved a competitive score of **0.6529**.  

## Installation 📚

To install OpenDeepSearch, run:

```bash
pip install -e . #you can also use: uv pip install -e .
pip install -r requirements.txt #you can also use: uv pip install -r requirements.txt
```

Note: you must have `torch` installed.
Note: using `uv` instead of regular `pip` makes life much easier!

### Using PDM (Alternative Package Manager) 📦

You can also use PDM as an alternative package manager for OpenDeepSearch. PDM is a modern Python package and dependency manager supporting the latest PEP standards.

```bash
# Install PDM if you haven't already
curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -

# Initialize a new PDM project
pdm init

# Install OpenDeepSearch and its dependencies
pdm install

# Activate the virtual environment
eval "$(pdm venv activate)"
```

PDM offers several advantages:
- Lockfile support for reproducible installations
- PEP 582 support (no virtual environment needed)
- Fast dependency resolution
- Built-in virtual environment management

## Setup

1. **Choose a Search Provider**:
   - **Option 1: Serper.dev**: Get **free 2500 credits** and add your API key.
     - Visit [serper.dev](https://serper.dev) to create an account.
     - Retrieve your API key and store it as an environment variable:

     ```bash
     export SERPER_API_KEY='your-api-key-here'
     ```

   - **Option 2: SearXNG**: Use a self-hosted or public SearXNG instance.
     - Specify the SearXNG instance URL when initializing OpenDeepSearch.
     - Optionally provide an API key if your instance requires authentication:

     ```bash
     export SEARXNG_INSTANCE_URL='https://your-searxng-instance.com'
     export SEARXNG_API_KEY='your-api-key-here'  # Optional
     ```

2. **Choose a Reranking Solution**:
   - **Quick Start with Jina**: Sign up at [Jina AI](https://jina.ai/) to get an API key for immediate use
   - **Self-hosted Option**: Set up [Infinity Embeddings](https://github.com/michaelfeil/infinity) server locally with open source models such as [Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct/tree/main)
   - For more details on reranking options, see our [Rerankers Guide](src/opendeepsearch/ranking_models/README.md)

3. **Set up LiteLLM Provider**:
   - Choose a provider from the [supported list](https://docs.litellm.ai/docs/providers/), including:
     - OpenAI
     - Anthropic
     - Google (Gemini)
     - OpenRouter
     - HuggingFace
     - Fireworks
     - And many more!
   - Set your chosen provider's API key as an environment variable:
   ```bash
   export <PROVIDER>_API_KEY='your-api-key-here'  # e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY
   ```
   - For OpenAI, you can also set a custom base URL (useful for self-hosted endpoints or proxies):
   ```bash
   export OPENAI_BASE_URL='https://your-custom-openai-endpoint.com'
   ```
   - You can set default LiteLLM model IDs for different tasks:
   ```bash
   # General default model (fallback for all tasks)
   export LITELLM_MODEL_ID='openrouter/google/gemini-2.0-flash-001'

   # Task-specific models
   export LITELLM_SEARCH_MODEL_ID='openrouter/google/gemini-2.0-flash-001'  # For search tasks
   export LITELLM_ORCHESTRATOR_MODEL_ID='openrouter/google/gemini-2.0-flash-001'  # For agent orchestration
   export LITELLM_EVAL_MODEL_ID='gpt-4o-mini'  # For evaluation tasks
   ```
   - When initializing OpenDeepSearch, you can specify your chosen model using the provider's format (this will override the environment variables):
   ```python
   search_agent = OpenDeepSearchTool(model_name="provider/model-name")  # e.g., "anthropic/claude-3-opus-20240229", 'huggingface/microsoft/codebert-base', 'openrouter/google/gemini-2.0-flash-001'
   ```

## Usage ️

You can use OpenDeepSearch independently or integrate it with **SmolAgents** for enhanced reasoning and code generation capabilities.

### Using OpenDeepSearch Standalone 🔍

```python
from opendeepsearch import OpenDeepSearchTool
import os

# Set environment variables for API keys
os.environ["SERPER_API_KEY"] = "your-serper-api-key-here"  # If using Serper
# Or for SearXNG
# os.environ["SEARXNG_INSTANCE_URL"] = "https://your-searxng-instance.com"
# os.environ["SEARXNG_API_KEY"] = "your-api-key-here"  # Optional

os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key-here"
os.environ["JINA_API_KEY"] = "your-jina-api-key-here"

# Using Serper (default)
search_agent = OpenDeepSearchTool(
    model_name="openrouter/google/gemini-2.0-flash-001",
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
```

### Running the Gradio Demo 🖥️

To try out OpenDeepSearch with a user-friendly interface, simply run:

```bash
python gradio_demo.py
```

This will launch a local web interface where you can test different search queries and modes interactively.

You can customize the demo with command-line arguments:

```bash
# Using Serper (default)
python gradio_demo.py --model-name "openrouter/google/gemini-2.0-flash-001" --reranker "jina"

# Using SearXNG
python gradio_demo.py --model-name "openrouter/google/gemini-2.0-flash-001" --reranker "jina" \
  --search-provider "searxng" --searxng-instance "https://your-searxng-instance.com" \
  --searxng-api-key "your-api-key-here"  # Optional
```

Available options:
- `--model-name`: LLM model to use for search
- `--orchestrator-model`: LLM model for the agent orchestrator
- `--reranker`: Reranker to use (`jina` or `infinity`)
- `--search-provider`: Search provider to use (`serper` or `searxng`)
- `--searxng-instance`: SearXNG instance URL (required if using `searxng`)
- `--searxng-api-key`: SearXNG API key (optional)
- `--serper-api-key`: Serper API key (optional, will use environment variable if not provided)
- `--openai-base-url`: OpenAI API base URL (optional, will use OPENAI_BASE_URL env var if not provided)

### Integrating with SmolAgents & LiteLLM 🤖⚙️

####

```python
from opendeepsearch import OpenDeepSearchTool
from smolagents import CodeAgent, LiteLLMModel
import os

# Set environment variables for API keys
os.environ["SERPER_API_KEY"] = "your-serper-api-key-here"  # If using Serper
# Or for SearXNG
# os.environ["SEARXNG_INSTANCE_URL"] = "https://your-searxng-instance.com"
# os.environ["SEARXNG_API_KEY"] = "your-api-key-here"  # Optional

os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key-here"
os.environ["JINA_API_KEY"] = "your-jina-api-key-here"

# Using Serper (default)
search_agent = OpenDeepSearchTool(
    model_name="openrouter/google/gemini-2.0-flash-001",
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

model = LiteLLMModel(
    "openrouter/google/gemini-2.0-flash-001",
    temperature=0.2
)

code_agent = CodeAgent(tools=[search_agent], model=model)
query = "How long would a cheetah at full speed take to run the length of Pont Alexandre III?"
result = code_agent.run(query)

print(result)
```
### ReAct agent with math and search tools 🤖⚙️

####
```python
from opendeepsearch import OpenDeepSearchTool
from opendeepsearch.wolfram_tool import WolframAlphaTool
from opendeepsearch.prompts import REACT_PROMPT
from smolagents import LiteLLMModel, ToolCallingAgent, Tool
import os

# Set environment variables for API keys
os.environ["SERPER_API_KEY"] = "your-serper-api-key-here"
os.environ["JINA_API_KEY"] = "your-jina-api-key-here"
os.environ["WOLFRAM_ALPHA_APP_ID"] = "your-wolfram-alpha-app-id-here"
os.environ["FIREWORKS_API_KEY"] = "your-fireworks-api-key-here"

model = LiteLLMModel(
    "fireworks_ai/llama-v3p1-70b-instruct",  # Your Fireworks Deepseek model
    temperature=0.7
)
search_agent = OpenDeepSearchTool(model_name="fireworks_ai/llama-v3p1-70b-instruct", reranker="jina") # Set reranker to "jina" or "infinity"

# Initialize the Wolfram Alpha tool
wolfram_tool = WolframAlphaTool(app_id=os.environ["WOLFRAM_ALPHA_APP_ID"])

# Initialize the React Agent with search and wolfram tools
react_agent = ToolCallingAgent(
    tools=[search_agent, wolfram_tool],
    model=model,
    prompt_templates=REACT_PROMPT # Using REACT_PROMPT as system prompt
)

# Example query for the React Agent
query = "What is the distance, in metres, between the Colosseum in Rome and the Rialto bridge in Venice"
result = react_agent.run(query)

print(result)
```

## Search Modes 🔄

OpenDeepSearch offers two distinct search modes to balance between speed and depth:

### Default Mode ⚡
- Uses SERP-based interaction for quick results
- Minimal processing overhead
- Ideal for single-hop, straightforward queries
- Fast response times
- Perfect for basic information retrieval

### Pro Mode 🔍
- Involves comprehensive web scraping
- Implements semantic reranking of results
- Includes advanced post-processing of data
- Slightly longer processing time
- Excels at:
  - Multi-hop queries
  - Complex search requirements
  - Detailed information gathering
  - Questions requiring cross-reference verification

## Acknowledgments 💡

OpenDeepSearch is built on the shoulders of great open-source projects:

- **[SmolAgents](https://huggingface.co/docs/smolagents/index)** 🤗 – Powers the agent framework and reasoning capabilities.
- **[Crawl4AI](https://github.com/unclecode/crawl4ai)** 🕷️ – Provides data crawling support.
- **[Infinity Embedding API](https://github.com/michaelfeil/infinity)** 🌍 – Powers semantic search capabilities.
- **[LiteLLM](https://www.litellm.ai/)** 🔥 – Used for efficient AI model integration.
- **Various Open-Source Libraries** 📚 – Enhancing search and retrieval functionalities.

## Citation

If you use `OpenDeepSearch` in your works, please cite it using the following BibTex entry:

```
@misc{alzubi2025opendeepsearchdemocratizing,
      title={Open Deep Search: Democratizing Search with Open-source Reasoning Agents},
      author={Salaheddin Alzubi and Creston Brooks and Purva Chiniya and Edoardo Contente and Chiara von Gerlach and Lucas Irwin and Yihan Jiang and Arda Kaz and Windsor Nguyen and Sewoong Oh and Himanshu Tyagi and Pramod Viswanath},
      year={2025},
      eprint={2503.20201},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.20201},
}
```


## Contact 📩

For questions or collaborations, open an issue or reach out to the maintainers.
