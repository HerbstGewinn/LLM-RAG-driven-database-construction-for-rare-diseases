#!/usr/bin/env python3
import os
import sys
from langchain_community.document_loaders import PyPDFLoader  
from langchain_text_splitters import RecursiveCharacterTextSplitter 
#from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
#import getpass
from langchain_core.prompts import PromptTemplate

def load_pdf(file_path):
    
    if not os.path.isfile(file_path):  
        print(f"ERROR: file '{file_path}' not found.")
        sys.exit(1)

    try:
       loader = PyPDFLoader(file_path)
       document = loader.load()  
       print("PDF loading successful")  
    except Exception as e:
        print(f"ERROR loading '{file_path}': {e}")
        sys.exit(1)
    
    text = ""
    for doc in document:
        text += doc.page_content
    return text
    
def splittext(text):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
    texts = text_splitter.split_text(text)
    print(f"Text successfully splitted into {len(texts)} chunks.")
    return [Document(page_content=chunk) for chunk in texts]
    

def vectorstore(textsplits):
    #os.environ["OPENAI_API_KEY"] = getpass.getpass()
    #embeddings = OpenAIEmbeddings()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    storage = InMemoryVectorStore(embedding=embeddings)
    storage.add_documents(textsplits)
    print("Vectorstore successfully created and documents added.")
    return storage

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag(retriever, question):
    #llm = ChatOpenAI(model="gpt-4o-mini")
    llm=ChatOllama(
    model="llama3.1:8b",
    )
    prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "Question: {question}\n"
        "Context: {context}\n"
        "Answer:"
    )
    )
    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    response=rag_chain.invoke(question)
    return response

    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No file path provided. Example: .\scriptname.py C:\path\\to\filename.pdf")
        sys.exit(1)

    pdf_file_path = sys.argv[1]  
    document = load_pdf(pdf_file_path)
    textsplits =splittext(document)
    storage=vectorstore(textsplits)
    retriever = storage.as_retriever()
    print("retriever successfully created")
    print("What do you want to know?")
    question= input(" ")
    response=rag(retriever, question)
    print(response)
    
 
    
    #print(f"Number of text splits: {len(textsplits)}")
    #print("First text split:", textsplits[0].page_content)
    #print("Second text split:", textsplits[1].page_content)
