# Local Document Query with LLM

This web application leverages the Ollama for querying documents through a user-friendly Gradio interface.

## Installation

1. Clone this repository.
2. Ensure Python 3.10+ is installed.
3. Install dependencies with `pip install -r requirements.txt`.
4. Download the Ollama from [Ollama's official download page](https://ollama.com/download).
5. Use the Ollama CLI to pull necessary models:
   - `ollama pull nomic-embed-text`
   - `ollama pull llama2`

## Usage

- Run `python app.py`.
- Navigate to the Gradio URL.
- Input URLs and a question for document queries.
- [demo](demo.ipynb) notebook with explanation
- [slides](demo.slides.html)


## Dependencies

- Chromadb  
    AI-native open-source embedding database
- Gradio  
    is an open-source Python package that allows you to quickly build a demo or web application for your machine learning model, API, or any arbitary Python function. You can then share a link to your demo or web application in just a few seconds using Gradio’s built-in sharing features. No JavaScript, CSS, or web hosting experience needed!
- langchain  
    is a framework designed to simplify the creation of applications using large language models (LLMs)
- langchain-core  
   is focused on providing base abstractions and the LangChain Expression Language, essential for creating and managing the components and interactions within the LangChain ecosystem.
- langchain-community  
   includes third-party integrations, expanding the framework's capabilities and interoperability with other tools and platforms.
- langchain  
   itself encompasses chains, agents, and retrieval strategies crucial for building the cognitive architecture of applications.

## License

GNU License - see LICENSE file for details.
