from pdfminer.high_level import extract_text
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
import os


# Set environment variable for protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


class PDFRetriever:
    def __init__(self, file_path, model="llama3.2:latest", chunk_size=1000, chunk_overlap=200):
        """
        Initializes the PDFDocumentAnalyzer with the given parameters.
        """
        self.file_path = file_path
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_db = None
        self.rag_chain = None
        self.document_text = None

    def load_pdf(self):
        """
        Loads the PDF file, processes its content, and creates a vector database.
        """
        try:
            self.document_text = extract_text(self.file_path)
            print(f"\nPDF loaded successfully: {self.file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF: {e}")
        
        document = Document(page_content=self.document_text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_documents([document])
        print(f"\nText split into {len(chunks)} chunks")

        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            collection_name="local-rag"
        )
        print("\nVector database created successfully...")

    def initialize_chain(self):
        """
        Sets up the LLM and retrieval chain for document analysis.
        """
        if not self.vector_db:
            raise RuntimeError("Vector database is not initialized. Call load_and_process_pdf first.")
   
        llm = ChatOllama(model=self.model, temperature=0)
       
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate 2
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}"""
        )

        retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_db.as_retriever(),
            llm=llm,
            prompt=query_prompt
        )
        
        class DiseaseTreatmentInfo(BaseModel):
            disease: str = Field(description="The name of the disease")
            treatment: str = Field(description="The suggested treatment")
            gene: List[str] = Field(description="List of associated genes")

        parser = JsonOutputParser(pydantic_object=DiseaseTreatmentInfo)

        prompt = PromptTemplate(
            template="""Answer the question based ONLY on the following context: 
            {context}. Extract information about disease, treatment, 
            and associated genes.\n{format_instructions}\n{question}\n""",
            input_variables=["question"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | parser
        )
        
        print("\nRetrieval and generation chain initialized successfully...")

    def analyze_document(self, query):
        """
        Analyzes the document based on the given query.
        """
        if not self.rag_chain:
            raise RuntimeError("RAG chain is not initialized. Call initialize_chain first.")
        
        print("\nProcessing your query, please wait...")
        response = self.rag_chain.invoke(query)
        self.vector_db.delete_collection()
        print("\nVector database deleted successfully")
        return response
    
        

    

  
    