import os
import hashlib
from typing import List, Dict, Any, Optional
import yaml
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class DocumentIndexer:
    """Class to handle document indexing operations for the RAG system."""



    def __init__(self, config_path: str):
        """Initialize the document indexer with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['embeddings']['model']
        )
        
        # define vector store type
        self.vector_store_type = self.config['vector_store']['type']
        self.persist_dir = self.config['vector_store']['persist_directory']
        
        # document processing parameters
        self.chunk_size = self.config['document_processing']['chunk_size']
        self.chunk_overlap = self.config['document_processing']['chunk_overlap']
        self.separators = self.config['document_processing']['separators']
        
        # data directory
        self.data_dir = self.config['data_directory']
        
        self.vector_store = None
    
    def _get_docs_hash(self) -> str:
        """Generate a hash of all documents to check if reindexing is needed."""
        hash_obj = hashlib.md5()
        for root, _, files in os.walk(self.data_dir):
            for file in sorted(files):
                if file.endswith('.pdf'):
                    path = os.path.join(root, file)
                    with open(path, 'rb') as f:
                        hash_obj.update(f.read())
        return hash_obj.hexdigest()
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from the data directory."""
        documents = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.pdf'):
                    path = os.path.join(root, file)
                    try:
                        loader = PyMuPDFLoader(path)
                        docs = loader.load()
                        for doc in docs:
                            # Ensure document metadata contains source
                            doc.metadata["source"] = path
                        documents.extend(docs)
                    except Exception as e:
                        print(f"Error loading document {path}: {e}")
        return documents
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        return text_splitter.split_documents(documents)
    
    def create_vector_store(self, documents: List[Dict[str, Any]]) -> Any:
        """Create a vector store from documents."""
        if self.vector_store_type == "FAISS":
            self.vector_store = FAISS.from_documents(
                documents, 
                self.embeddings
            )
            self.vector_store.save_local(self.persist_dir)
        elif self.vector_store_type == "Chroma":
            self.vector_store = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=self.persist_dir
            )
            self.vector_store.persist()
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
        
        return self.vector_store
    
    def load_vector_store(self) -> Optional[Any]:
        """Load an existing vector store if available."""
        if os.path.exists(self.persist_dir):
            if self.vector_store_type == "FAISS":
                self.vector_store = FAISS.load_local(
                    self.persist_dir, 
                    self.embeddings,
                    allow_dangerous_deserialization= True
                )
            elif self.vector_store_type == "Chroma":
                self.vector_store = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings
                )
            return self.vector_store
        return None
    
    def index_documents(self, force_rebuild: bool = False) -> Any:
        """Index documents and return the vector store."""
        docs_hash = self._get_docs_hash()
        hash_file = f"{self.persist_dir}_hash.txt"
        
        if not force_rebuild and os.path.exists(self.persist_dir) and os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            if stored_hash == docs_hash:
                return self.load_vector_store()
        
        documents = self.load_documents()
        chunked_documents = self.split_documents(documents)
        
        # Create vector store
        vector_store = self.create_vector_store(chunked_documents)
        
        with open(hash_file, 'w') as f:
            f.write(docs_hash)
        
        return vector_store