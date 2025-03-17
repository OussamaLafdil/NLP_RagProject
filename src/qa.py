from typing import Dict, List, Any, Optional
import yaml
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

class QASystem:
    """Class to handle question answering operations for the RAG system."""
    
    def __init__(self, vector_store, config_path: str):
        """Initialize the QA system with a vector store and configuration."""
        self.vector_store = vector_store
        self.retriever = None
        self.llm = None
        self.chain = None
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self._initialize_llm()
        
        if self.vector_store:
            self.retriever = self.vector_store.as_retriever()
            
            self._create_chain()
    
    def _initialize_llm(self):
        """Initialize the language model based on configuration."""
        llm_provider = self.config['llm']['provider']
        
        if llm_provider == "huggingface":
            # HuggingFace for LLM
            self.llm = HuggingFaceEndpoint(
                repo_id=self.config['llm']['model'],
                task="text-generation",
                max_new_tokens=2000
            )
        elif llm_provider == "google" :
            # Gemini 
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.7,
                max_tokens=2000,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    def _create_chain(self):
        """Create the question-answering chain."""
        qa_prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant that answers questions based on the provided context.
        You are an expert in Pytorch coding.
        Generate Pytorch code when it is relevant.        
        Context:
        {context}
        
        Question: {input}
        
        Answer the question and provide code example"
        """)
        
        # Create document template
        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="{page_content}\nSource: {source}"
        )
        
        # Create document chain
        doc_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=qa_prompt,
            document_prompt=document_prompt,
            document_variable_name="context"
        )
        
        # Create retrieval chain
        self.chain = create_retrieval_chain(self.retriever, doc_chain)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the RAG system.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary containing the answer and sources
        """
        if not self.chain:
            raise ValueError("QA chain not initialized")
        
        # get response from chain
        response = self.chain.invoke({"input": question})
        
        # get sources
        sources = []
        if "context" in response:
            for doc in response["context"]:
                if doc.metadata and "source" in doc.metadata:
                    if doc.metadata["source"] not in sources:
                        sources.append(doc.metadata["source"])
        
        return {
            "answer": response["answer"],
            "sources": sources
        }