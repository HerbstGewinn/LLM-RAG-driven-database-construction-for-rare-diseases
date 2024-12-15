from pdfminer.high_level import extract_text
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import LLMChain
#from langchain_core.output_parsers import JsonOutputParser
#from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from graph_state import GraphState
import json
import os
import requests
from dotenv import load_dotenv
#from IPython.display import Image, display


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

class PDFRetriever:
    def __init__(self, file_path, model="llama3.2:latest", chunk_size=1000, chunk_overlap=200):
        
        self.file_path = file_path
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_db = None
        self.document_text = None
        self.retriever = None

        self.workflow = StateGraph(GraphState)

        #Define the nodes
        self.workflow.add_node("load_pdf", self.load_pdf)
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("hallucination_check", self.hallucination_check)
        self.workflow.add_node("validate_answer", self.validate_answer)
        self.workflow.add_node("adjust_query", self.adjust_query)

        #Build Graph 
        self.workflow.set_conditional_entry_point(
            self.pdf_is_loaded,
            {
                "load_pdf" : "load_pdf",
                "retrieve" : "retrieve",
            },
        )
        self.workflow.add_edge("retrieve", "hallucination_check")
        self.workflow.add_edge("adjust_query", "retrieve")
        self.workflow.add_conditional_edges(
            "hallucination_check",
            self.is_hallucinating,
            {
                "no hallucination" : "validate_answer",
                "hallucination" : "adjust_query",
                "max retries" : END,
            },
        )
        self.workflow.add_conditional_edges(
            "validate_answer",
            self.is_valid,
            {
                "valid" : END,
                "not valid" : "adjust_query",
                "max retries" : END,
            },
        )

        load_dotenv()
        user = os.getenv("ollama_user")
        password = os.getenv("ollama_pw")

        # Authentication details
        protocol = "https"
        hostname = "chat.cosy.bio"
        host = f"{protocol}://{hostname}"
        auth_url = f"{host}/api/v1/auths/signin"
        self.api_url = f"{host}/ollama"
        account = {
            'email': user,
            'password': password
        }
        auth_response = requests.post(auth_url, json=account)

        jwt = auth_response.json()["token"]
        self.headers = {"Authorization": "Bearer " + jwt}

    def load_pdf(self):
       
        try:
            self.document_text = extract_text(self.file_path)
            print(f"---PDF LOADED SUCCESSFULLY: {self.file_path}---")
        except Exception as e:
            raise RuntimeError(f"---FAILED TO LOAD PDF: {e}---")
        
        document = Document(page_content=self.document_text)
        text_splitter = RecursiveCharacterTextSplitter(
                                                chunk_size=self.chunk_size, 
                                                chunk_overlap=self.chunk_overlap
                                                    )
        chunks = text_splitter.split_documents([document])
        print(f"---TEXT SPLIT INTO {len(chunks)} CHUNKS---")

        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(base_url=self.api_url, 
                                       model="nomic-embed-text", 
                                       client_kwargs={"headers": self.headers}),
            collection_name="local-rag"
        )
        print("---VECTOR DATABASE CREATED SUCCESSFULLY---")

    def initialize_chain(self):
       
        if not self.vector_db:
            raise RuntimeError("Vector database is not initialized. Call load_and_process_pdf first.")
   
        self.llm = ChatOllama(model=self.model, temperature=0, 
                          base_url=self.api_url, 
                          client_kwargs={"headers": self.headers}, 
                          format="json",
                          num_ctx=25000)
       
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate 3
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}"""
        )

        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_db.as_retriever(),
            llm=self.llm)
        
        print("---RETRIEVAL INITIALIZED SUCCESSFULLY---")

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def delete_db(self):
        self.vector_db.delete_collection()
        print("\n---VECTOR DATABASE DELETED SUCCESSFULLY---")

    def pdf_is_loaded(self, state):
        if self.vector_db:
            return "retrieve"
        return "load_pdf"

    def adjust_query(self, state):
        return {"question" : f"""Original question: {state["original_question"]} 
            \nIn a previous iteration, your answer to the original question above was:
            {state['generation']}.\n
            This answer was not acceptable for the following reason: 
            {state['current_explanation']}. 

            Please provide a revised answer that addresses this feedback. 
            Make sure your response is accurate, relevant, and complete. 
            Pay special attention to:
            - Correctly interpret the question
            - Avoiding previous errors
            - Addressing the specific issues highlighted above.
            """}

    def hallucination_check(self, state):
        instructions = """ 

        You are a teacher grading whether an answer is based only on the given facts.

        You will be given FACTS and a LLM ANSWER.

        Here is the grading criteria to follow:

        1. Ensure the LLM ANSWER does not contain "hallucinated" information outside the scope of the FACTS. 
        - Information that is directly supported by the FACTS is acceptable.
        - If the LLM ANSWER includes multiple details or examples, ensure that all of them are relevant to the context provided by the FACTS.

        2. Ensure the LLM ANSWER is short, precise, and focused. 
        - Answers can include multiple relevant aspects as long as they are clearly and concisely presented.

        3. Grading:

        - A score of "yes" means the LLM's answer fully meets all criteria: 
            - It is based only the FACTS and does not introduce unrelated or incorrect information.
        
        - A score of "no" means the LLM's answer does not meet the criteria:
            - It includes hallucinated or unrelated information.

        4. Explain your reasoning in a step-by-step manner:
        - Highlight whether each aspect of the LLM ANSWER aligns with the FACTS.
        - Clearly indicate why specific parts of the answer are correct or incorrect.
        """

        prompt = """FACTS: \n\n {documents} \n\n LLM ANSWER: {generation}.

        You MUST strictly follow this JSON format:
            {{
                "binary_score" : "string",
                "explanation" : "string"
            }}

        There are two keys, binary_score is 'yes' or 'no' score to indicate the LLM ANSWER is grounded in the FACTS.
        The second key, explanation, gives reasoning for the provided binar_score.
        
        """
        answers = ", ".join(item for item in state["generation"])
        prompt_formatted = prompt.format(documents=self.doc_txt, generation=answers)
        result = self.llm.invoke([SystemMessage(content=instructions)] + [HumanMessage(content=prompt_formatted)])
        result_content = json.loads(result.content)
        return {
            "is_hallucinating" : result_content["binary_score"],
            "current_explanation" : result_content["explanation"]
            }
                    
    def is_hallucinating(self, state):
        max_retries = state.get("max_retries", 3)
        result = state["is_hallucinating"]
        if result == "yes":
            print("---DECISION: ANSWER IS GROUNDED IN THE DOCUMENT---")
            return "no hallucination"
        
        elif state["loop_step"] <= max_retries:
            print("---DECISION: ANSWER IS NOT GROUNDED IN THE DOCUMENT---")
            print(f"---EXPLANATION: {state["current_explanation"]}")
            print("Adjusting the query...")
            print("Starting new attempt...")
            return "hallucination"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"

    def validate_answer(self, state):
        instructions = """You are a teacher grading a quiz.
        
        You will be given a QUESTION and a LLM ANSWER. 
        
        Here is the grade criteria to follow:
        
        (1) The LLM ANSWER helps to answer the QUESTION 

        (2) Ensure the LLM ANSWER is SHORT and PRECISE and without unnecessary additional information.
        
        Score:
        
        A score of yes means that the llm's answer meets all of the criteria. This is the highest (best) score.

        A score of no means that the llm's answer does not meet all of the criteria. This is the lowest possible score you can give.

        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

        Avoid simply stating the correct answer at the outset.
        """

        prompt = """QUESTION: \n\n {question} \n\n LLM ANSWER: {generation}.

        You MUST strictly follow this JSON format:
            {{
                "binary_score" : "string",
                "explanation" : "string"
            }}

        There are two keys, binary_score is 'yes' or 'no' score to indicate whether the LLM ANSWER mets the criteria.
        The second key, explanation, gives reasoning for the provided binar_score.
        
        """
        answers = ", ".join(item for item in state["generation"])
        query = state["original_question"]
        prompt_formatted = prompt.format(question=query, generation=answers)
        result = self.llm.invoke([SystemMessage(content=instructions)] + [HumanMessage(content=prompt_formatted)])
        result_content = json.loads(result.content)
        return {"is_valid" : result_content["binary_score"], 
                "current_explanation" : result_content["explanation"]
                }
        
    def is_valid(self, state):
        max_retries = state.get("max_retries", 3)
        result = state["is_valid"]
        if result == "yes":
            print("---DECISION: ANSWER ADDRESSES QUESTION---")
            return "valid"
        
        elif state["loop_step"] <= max_retries:
            print("---DECISION: ANSWER DOES NOT ADDRESS QUESTION---")
            print(f"---EXPLANATION: {state["current_explanation"]}")
            print("Adjusting the query...")
            print("Starting new attempt...")
            return "not valid"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"

    def retrieve(self, state):
        
        rag_prompt = """Answer the question based ONLY on the following context: 
            {context}. 
            
            Think carefully about the above context.
            
            Now, review the user question:
            
            {question}
            
            Your answer should be SHORT and PRECISE and without additional information.
            You MUST strictly follow this JSON format:
            {{"answer" : ["string"]}}
            
            """
        
        print("\nProcessing your query, please wait...")
        query = state["question"]
        loop_step = state.get("loop_step", 0)
        docs = self.retriever.invoke(query)
        self.doc_txt = self.format_docs(docs)
        rag_prompt_formattet = rag_prompt.format(context=self.doc_txt, question=query)
        generation = self.llm.invoke([HumanMessage(content=rag_prompt_formattet)])
        generation_content = json.loads(generation.content)
        answer = generation_content["answer"]
        print(f"\nCurrent answer: {answer}")
        return {
            "generation" : answer,
            "loop_step" : loop_step + 1
            }

    def analyze_document(self, query):
        graph = self.workflow.compile()
        # Graph speichern
        #with open("graph.png", "wb") as f:
            #f.write(image_data)
        inputs = {"question" : query, "original_question" : query, "max_retries" : 3}    
        for event in graph.stream(inputs, stream_mode="values"):
            current_state = event
        return current_state['generation']
        
    
        

    

  
    