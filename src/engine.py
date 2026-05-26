import os
from typing import List, Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class DocumentManager:
    def __init__(self, upload_dir: str = "data/uploads"):
        self.upload_dir = upload_dir
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False,
        )

    def load_and_split(self, file_path: str):
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")
            
            docs = loader.load()
            return self.text_splitter.split_documents(docs)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def load_directory(self, path: str):
        """Load all compatible files from a directory."""
        if not os.path.exists(path):
            return []
            
        all_chunks = []
        for file in os.listdir(path):
            if file.endswith((".pdf", ".txt")):
                file_path = os.path.join(path, file)
                chunks = self.load_and_split(file_path)
                all_chunks.extend(chunks)
        return all_chunks

class RAGEngine:
    def __init__(self, api_key: Optional[str] = None, provider: str = "openai"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm = self._initialize_llm(api_key, provider)
        self.vector_store = None

    def _initialize_llm(self, api_key: Optional[str], provider: str):
        if not api_key or api_key.strip() == "":
            return None
        
        try:
            if provider == "openai":
                return ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)
            elif provider == "groq":
                return ChatGroq(api_key=api_key, model_name="mixtral-8x7b-32768")
        except Exception as e:
            print(f"LLM initialization error: {e}")
            return None
        return None

    def create_vector_store(self, chunks):
        if not chunks:
            return None
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        return self.vector_store

    def get_response(self, query: str) -> Tuple[str, List]:
        if not self.llm:
            return "API Key is missing or invalid. Please check your settings.", []
        if not self.vector_store:
            return "No documents have been indexed yet. Please upload files to begin.", []

        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            
            system_prompt = (
                "You are an expert Policy Analyst assistant. "
                "Use the provided context to answer the user's question accurately. "
                "If the answer is not in the context, state that you don't have enough information. "
                "Structure your response with bullet points if applicable for clarity. "
                "\n\n"
                "Context: {context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            response = rag_chain.invoke({"input": query})
            return response["answer"], response["context"]
        except Exception as e:
            return f"An error occurred while generating a response: {str(e)}", []
