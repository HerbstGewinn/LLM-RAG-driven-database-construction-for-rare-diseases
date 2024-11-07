
# Imports
# from langchain_community.document_loaders import UnstructuredPDFLoader
import argparse

from pdfminer.high_level import extract_text

from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Suppress warnings
#import warnings
#warnings.filterwarnings('ignore')

# Set environment variable for protobuf
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Load PDF
def load_pdf(file_path):
    try:
        data = extract_text(file_path)
        print(f"PDF loaded successfully: {file_path}")
    except Exception as e:
        print(f"{file_path} can not be opened: {e}")
        exit(1)
    document = Document(page_content=data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents([document])
    print(f"Text split into {len(chunks)} chunks")
    vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="local-rag"
    )
    print("Vector database created successfully")
    return vector_db

def load_llm(local_model, vector_db):
    # Set up LLM and retrieval
    llm = ChatOllama(model=local_model)
    # Query prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    # Set up retriever
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    # RAG prompt template
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    # Create chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def main():
    parser = argparse.ArgumentParser(description="Program for document (pdf) analysis with different LLMs.")
    # Choose model
    parser.add_argument(
        '-m','--model', 
        type=str, 
        default='llama3.1:latest',
        help="Which model do you want to use? (Standard: llama3.1:latest)"
    )

    # Choose file
    parser.add_argument(
        '-f','--file', 
        type=str, 
        required=True,
        help="Path to pdf file for analysis."
    )

    args = parser.parse_args()
    file_path = args.file
    local_model = args.model

    vector_db = load_pdf(file_path)
    rag_chain = load_llm(local_model, vector_db)
    first_query = True
    while(True):
        if first_query:
            first_query = False
            print("\n\nWhat do you want to know about the uploaded document?")
            query = input("Type in a query and press [enter] or type [exit]/[q] to quit.\n\n")
        else:
            query = input("Type in another query and press [enter] or type [exit]/[q] to quit.\n\n")
        if query in ["exit", "q"]:
            print("Exiting program ...")
            vector_db.delete_collection()
            print("\nVector database deleted successfully")
            exit(0)
        print("Document is being analysed, please wait ...")
        response = rag_chain.invoke(query)
        print(f"\n{local_model} response:\n")
        print(response)
    

if __name__ == "__main__":
    main()

