#!/usr/bin/env python

import argparse
import os
import sys
import json
from datetime import datetime
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.indexer import DocumentIndexer
from src.searcher import DocumentSearcher
from src.qa import QASystem
from src.evaluator import RAGEvaluator
from src.chat import ChatSystem
from src.utils import format_sources, ensure_directory_exists

def index_documents(args):
    """Index documents in the data directory."""
    indexer = DocumentIndexer(args.config)
    vector_store = indexer.index_documents(force_rebuild=args.force_rebuild)
    print(f"Documents indexed successfully. Vector store saved to {indexer.persist_dir}")
    return 0

def search_documents(args):
    """Search for documents matching a query."""
    indexer = DocumentIndexer(args.config)
    vector_store = indexer.load_vector_store()
    
    if not vector_store:
        print("Vector store not found. Please index documents first.")
        return 1
    
    searcher = DocumentSearcher(vector_store, args.config)
    results = searcher.search(args.query, k=args.limit)
    
    print(f"\nSearch Results for: '{args.query}'\n")
    print(f"Found {len(results)} results\n")
    
    for i, result in enumerate(results):
        print(f"Result {i+1}: [Score: {result['score']:.4f}]")
        print(f"Source: {result['metadata']['source']}")
        print(f"Content: {result['content'][:400]}...")  
        print("-" * 80)
    
    return 0

def ask_question(args):
    """Ask a question to the RAG system."""
    indexer = DocumentIndexer(args.config)
    vector_store = indexer.load_vector_store()
    
    if not vector_store:
        print("Vector store not found. Please index documents first.")
        return 1
    
    qa_system = QASystem(vector_store, args.config)
    response = qa_system.answer_question(args.question)
    
    print("\nQuestion:")
    print(args.question)
    print("\nAnswer:")
    print(response["answer"])
    print("\nSources:")
    print(format_sources(response["sources"]))
    
    return 0

def evaluate_system(args):
    """Evaluate the RAG system using test questions."""
    indexer = DocumentIndexer(args.config)
    vector_store = indexer.load_vector_store()
    
    if not vector_store:
        print("Vector store not found. Please index documents first.")
        return 1
    
    qa_system = QASystem(vector_store, args.config)
    evaluator = RAGEvaluator(args.config)
    
    # Load test questions
    if not os.path.exists(args.test_file):
        print(f"Test file not found: {args.test_file}")
        return 1
    
    with open(args.test_file, 'r') as file:
        test_questions = [line.strip() for line in file if line.strip()]
    
    print(f"Running evaluation on {len(test_questions)} test questions...")
    results = evaluator.benchmark_system(qa_system, test_questions)
    
    # Save results
    ensure_directory_exists("evaluation_results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"evaluation_results/eval_{timestamp}.json"
    
    with open(result_file, 'w') as file:
        json.dump(results, file, indent=2)
    
    print(f"Evaluation completed. Results saved to {result_file}")
    return 0

def chat_interactive(args):
    """Start an interactive chat session."""
    indexer = DocumentIndexer(args.config)
    vector_store = indexer.load_vector_store()
    
    if not vector_store:
        print("Vector store not found. Please index documents first.")
        return 1
    
    chat_system = ChatSystem(vector_store, args.config)
    user_id = args.user_id or "default_user"
    
    print("\nInteractive RAG Chat (type 'exit' to quit, 'clear' to clear history)\n")
    
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'exit':
                print("\nExiting chat. Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                chat_system.clear_history(user_id)
                print("\nChat history cleared.")
                continue
            
            response = chat_system.generate_response(user_id, user_input)
            
            print("\nAssistant: " + response["answer"])
            
            if args.show_sources and response["sources"]:
                print("\nSources:")
                print(format_sources(response["sources"]))
        
        except KeyboardInterrupt:
            print("\n\nExiting chat. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="RAG System CLI")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument(
        "--force-rebuild", 
        action="store_true",
        help="Force rebuilding the vector store"
    )
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", 
        type=int, 
        default=5,
        help="Maximum number of results to return"
    )
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", type=str, help="Question to ask")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the RAG system")
    eval_parser.add_argument(
        "--test-file", 
        type=str, 
        required=True,
        help="Path to file containing test questions (one per line)"
    )
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument(
        "--user-id", 
        type=str, 
        default=None,
        help="User ID for chat history"
    )
    chat_parser.add_argument(
        "--show-sources", 
        action="store_true",
        help="Show document sources in chat responses"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # execute based on command
    if args.command == "index":
        return index_documents(args)
    elif args.command == "search":
        return search_documents(args)
    elif args.command == "ask":
        return ask_question(args)
    elif args.command == "evaluate":
        return evaluate_system(args)
    elif args.command == "chat":
        return chat_interactive(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())