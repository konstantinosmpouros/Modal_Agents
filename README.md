# Multi-Agent LLM System with Modal & LangGraph

This project showcases a scalable multi-agent system powered by large language models (LLMs), served via [Modal.com](https://modal.com) and orchestrated using [LangGraph](https://www.langchain.com/langgraph). Each agent is designed to handle a specialized task and communicates with others through a graph-based workflow.

## Features

- 🔗 **LangGraph Orchestration**: Agents work in a dynamic, directed graph to handle multi-step tasks.
- ⚡ **Modal-Hosted LLMs**: Efficient, scalable, and low-latency LLM inference with Modal.
- 🧠 **Multi-Agent Design**: Separation of responsibilities across agents (e.g., reasoning, retrieval, summarization).
- 🧪 **Use Case Ready**: Easily adaptable for chatbots, document processing, RAG pipelines, and more.

## Components

- `RAG/`: Modules related to retrieval-augmented generation.
- `utils/`: Helper functions and shared utilities.
- `Multi_Agents/`: LangGraph workflows defining agent interactions. Definitions of each agent and their logic.
- `APIs/`: Modal deployment scripts for serving LLMs.
- `__main__.py`: Entry point to run the multi-agent system.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Modal & Graphviz Setup

To install modal & graphviz libraries

```python
pip install modal
sudo apt-get install graphviz
```

- Make sure to add the Graphviz bin/ directory to your system's PATH variable after installation. Installation setup, [here](https://graphviz.org/download/).

To setup the connection between your device and modal

```python
modal setup

or 

python -m modal setup
```

## How to Run

After setting up Modal and installing the dependencies along with the Graphviz, run the system from the root directory:

```bash
python __main__.py

or

python .
```

you can also run the project from any other directory with the following command:

```bash
python path/Modal_Agents
```
