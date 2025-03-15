from typing import List, Dict, Any, Optional
import yaml

class DocumentSearcher:
    """Class to handle document search operations for the RAG system."""
    
    def __init__(self, vector_store, config_path: str):
        """Initialize the document searcher with a vector store and configuration."""
        self.vector_store = vector_store
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of documents with scores
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        # get documents with scores
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs_with_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
        
        return results