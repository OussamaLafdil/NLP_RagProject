from typing import Dict, List, Any, Optional
import yaml
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

class ChatSystem:
    """Class to handle chat operations for the RAG system."""
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

    def __init__(self, vector_store, config_path: str):
        """Initialize the chat system with a vector store and configuration."""
        self.vector_store = vector_store
        self.user_histories = {}  # Store chat histories by user ID
        self.llm = None
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the language model based on configuration."""
        llm_provider = self.config['llm']['provider']
        
        if llm_provider == "huggingface":
            #HuggingFace Endpoint
            self.llm = HuggingFaceEndpoint(
                repo_id=self.config['llm']['model'],
                task="text-generation",
                max_new_tokens=512
            )
        elif llm_provider == "google" :
            # Gemini for llm
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.7,
                max_tokens=2000,
            )


        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    def get_user_memory(self, user_id: str) -> List[Dict]:
        """Gets or initializes the chat history for a user."""
        if user_id not in self.user_histories:
            self.user_histories[user_id] = []
        return self.user_histories[user_id]
    
    def _prepare_chat_history(self, user_id: str) -> List:
        """Convert the stored messages to a list of AIMessage and HumanMessage objects."""
        history = self.get_user_memory(user_id)
        formatted_history = []
        for message in history:
            if message["role"] == "user":
                formatted_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                formatted_history.append(AIMessage(content=message["content"]))
        return formatted_history
    
    def generate_response(self, user_id: str, question: str) -> Dict[str, Any]:
        """
        Generate a response to a question using chat history.
        
        Args:
            user_id: The user ID for tracking conversation history
            question: The user's question
            
        Returns:
            Dictionary with answer and sources
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        # Get chat history
        memory = self.get_user_memory(user_id)
        chat_history = self._prepare_chat_history(user_id)
        

        history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, 
            self.vector_store.as_retriever(), 
            history_aware_retriever_prompt
        )
        
        #  chat template
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that answers questions based on the documents provided.
            Use only the information from the provided context to answer the question.
            If you cannot find the answer in the context, say "I don't have enough information to answer this question."
            Be concise, accurate, and helpful.
            
            Context:
            {context}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        #  document template
        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="{page_content}\nSource: {source}"
        )
        
        # documnt chain
        doc_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=chat_prompt,
            document_prompt=document_prompt,
            document_variable_name="context"
        )
        
        # retrieval chain
        retrieval_chain = create_retrieval_chain(history_aware_retriever, doc_chain)
        
        # Get response
        response = retrieval_chain.invoke({
            "input": question, 
            "chat_history": chat_history
        })
        
        # Extract sources
        sources = []
        if "context" in response:
            for doc in response["context"]:
                if doc.metadata and "source" in doc.metadata:
                    if doc.metadata["source"] not in sources:
                        sources.append(doc.metadata["source"])
        

        memory.append({"role": "user", "content": question})
        memory.append({"role": "assistant", "content": response["answer"]})
        
        return {
            "answer": response["answer"],
            "sources": sources
        }
    
    def clear_history(self, user_id: str) -> None:
        """Clear the chat history for a user."""
        if user_id in self.user_histories:
            del self.user_histories[user_id]