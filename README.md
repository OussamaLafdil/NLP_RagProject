# RAG System

This project implements a Retrieval Augmented Generation (RAG) system for question answering over document collections.

## Features

- Document indexing with vector storage
- Document search with relevance scores
- Question answering using LLM-based RAG
- System evaluation mechanisms
- Interactive chat functionality with memory

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/OussamaLafdil/NLP_RagProject
   cd rag-system
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure your environment:
   - Create a `.env` file with your API keys if needed
   - Adjust the `config.yaml` file to your needs

## Usage

### CLI Commands

The system provides several command-line commands:

#### Indexing Documents

```
python cli.py index [--force-rebuild]
```

This command indexes the documents in the data directory and creates a vector store.

#### Searching Documents

```
python cli.py search "your search query" [--limit N]
```

Search for documents matching a query and display the results.

#### Asking Questions

```
python cli.py ask "your question here"
```

Ask a question to get an answer based on the indexed documents.

#### Evaluating the System

```
python cli.py evaluate --test-file questions.txt
```

Evaluate the system using a list of test questions.

#### Interactive Chat

```
python cli.py chat [--user-id USER_ID] [--show-sources]
```

Start an interactive chat session with the RAG system.

## Configuration

The system is configured through the `config.yaml` file. Key configuration options include:

- Vector store type and settings
- Document processing parameters
- Embedding model selection
- LLM provider and model selection

## Adding Documents

Place your PDF documents in the `data/` directory. The system will automatically index them when you run the index command.

## Evaluation

To evaluate the system, create a text file with test questions (one per line) and use the `evaluate` command. Results will be saved in the `evaluation_results/` directory.

## Project Structure

```
project_root/
├── src/                    # Core RAG system code
│   ├── __init__.py
│   ├── indexer.py          # Document indexing
│   ├── searcher.py         # Vector search
│   ├── qa.py               # Q&A system
│   ├── evaluator.py        # Evaluation system
│   ├── chat.py             # Chat system
│   └── utils.py            # Helper functions
├── data/                   # PDF documents
├── cli.py                  # Command line interface
├── config.yaml             # Configuration file
└── requirements.txt        # Dependencies
```

