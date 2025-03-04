from pdfminer.high_level import extract_text
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
#from langchain_core.output_parsers import JsonOutputParser
#from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from graph_state import GraphState
import threading
import json
import os
#from sentence_transformers import CrossEncoder

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

class InvokeWithTimeout(threading.Thread):
    def __init__(self, target, args=(), kwargs=None):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.result = None
        self.exception = None

    def run(self):
        try:
            self.result = self.target(*self.args, **self.kwargs)
        except Exception as e:
            self.exception = e

class PDFRetriever:
    def __init__(self, file_path, chunk_size=1000, chunk_overlap=200):
        
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_db = None
        self.document_text = None
        self.retriever = None
        self.doc_txt = None
        #self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') 
        self.workflow = StateGraph(GraphState)

        #Define the nodes
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("evaluate_answer", self.evaluate_answer)
        self.workflow.add_node("adjust_query", self.adjust_query)

        #Build Graph 
        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "evaluate_answer")
        self.workflow.add_edge("adjust_query", "retrieve")
        self.workflow.add_conditional_edges(
            "evaluate_answer",
            self.evaluate_state,
            {
                "valid" : END,
                "hallucination" : "adjust_query",
                "not valid" : "adjust_query",
                "max retries" : END,
                "timeout" : END,
            },
        )
    
    def exclude_reference_part(self, text):
    
        normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized_text.split("\n")
        result = []
        for line in lines:
            if line.strip().lower() in ["references", "bibliography", "acknowledgments", "acknowledgements"]:
                break  
            result.append(line)
        
        return "\n".join(result).strip()

    def load_pdf(self, embedding_model, keep_references):
       
        try:
            self.document_text = extract_text(self.file_path)
            print(f"---PDF LOADED SUCCESSFULLY: {self.file_path}---")
        except Exception as e:
            raise RuntimeError(f"---FAILED TO LOAD PDF: {e}---")
        
        if keep_references == "False":
            self.document_text = self.exclude_reference_part(self.document_text)

        document = Document(page_content=self.document_text)
        text_splitter = RecursiveCharacterTextSplitter(
                                                chunk_size=self.chunk_size, 
                                                chunk_overlap=self.chunk_overlap
                                                    )
        chunks = text_splitter.split_documents([document])
        print(f"---TEXT SPLIT INTO {len(chunks)} CHUNKS---")

        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name="local-rag"
        )
        print("---VECTOR DATABASE CREATED SUCCESSFULLY---")

    def initialize_chain(self, llm):
       
        if not self.vector_db:
            raise RuntimeError("Vector database is not initialized. Call load_and_process_pdf first.")
   
        self.custom_llm = llm
       
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_db.as_retriever(),
            llm=self.custom_llm)
        
        print("---RETRIEVAL INITIALIZED SUCCESSFULLY---")

    def invoke_with_timeout(self, func, timeout, *args, **kwargs):
        thread = InvokeWithTimeout(target=func, args=args, kwargs=kwargs)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError("Operation timed out")
        if thread.exception:
            raise thread.exception
        return thread.result

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def delete_db(self):
        if self.vector_db:
            self.vector_db.delete_collection()
            print("\n---VECTOR DATABASE DELETED SUCCESSFULLY---")

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
    
    def evaluate_answer(self, state):

        if state["timeout"]:
            return
        
        instructions = """
        You are an impartial evaluator tasked with:
        
        1. Checking if the given ANSWER aligns strictly with the provided FACTS and contains no hallucinated information.
        2. Determining if the ANSWER adequately addresses the given QUESTION. 
        
        Scoring criteria:
        
        1. **Hallucination Check**:
            - A score of "yes" means the ANSWER is based solely on the FACTS and contains no hallucinated or unrelated information.
            - A score of "no" means the ANSWER contains hallucinated or unsupported information.
        
        2. **Answer Validation**:
            - A score of "yes" means the ANSWER directly addresses the QUESTION and avoids irrelevant information.
            - A score of "no" means the ANSWER does not adequately address the QUESTION.
        
        Provide your reasoning step by step for each evaluation.
        Return the results in this JSON format:
        {
            "hallucination_score": "string",
            "validation_score": "string",
            "explanation": "string"
        }
        """

        prompt = """
        FACTS: 
        {documents}

        QUESTION:
        {question}

        ANSWER:
        {generation}

        You MUST strictly follow this JSON format:
        {{
            "hallucination_score": "string",
            "validation_score": "string",
            "explanation": "string"
        }}
        """

        documents = self.doc_txt
        question = state["original_question"]
        answers = state["generation"]
        prompt_formatted = prompt.format(documents=documents, question=question, generation=answers)

        result = self.custom_llm.invoke(
            [SystemMessage(content=instructions)] + [HumanMessage(content=prompt_formatted)]
        )

        try:
            result_content = json.loads(result.content)
        except json.JSONDecodeError:
            print("Invalid JSON output. Raw content:", result.content)
            result_content = {
                "hallucination_score": "no",
                "validation_score": "no",
                "explanation": "The output was not JSON-compliant."
            }

        return {
            "is_hallucinating": result_content["hallucination_score"],
            "is_valid": result_content["validation_score"],
            "current_explanation": result_content["explanation"]
        }
    
    def evaluate_state(self, state):

        if state["timeout"]:
            return "timeout"
       
        max_retries = state.get("max_retries", 3)
        for key in ["is_hallucinating", "is_valid"]:
            result = state[key]

            if result == "yes":
                decision_message = (
                    "---DECISION: ANSWER IS GROUNDED IN THE DOCUMENT---"
                    if key == "is_hallucinating"
                    else "---DECISION: ANSWER ADDRESSES QUESTION---"
                )
                print(decision_message)

            elif state["loop_step"] <= max_retries:
                decision_message = (
                    "---DECISION: ANSWER IS NOT GROUNDED IN THE DOCUMENT---"
                    if key == "is_hallucinating"
                    else "---DECISION: ANSWER DOES NOT ADDRESS QUESTION---"
                )
                print(decision_message)
                print(f"---EXPLANATION: {state['current_explanation']}")
                print("Adjusting the query...")
                print("Starting new attempt...")
                return "hallucination" if key == "is_hallucinating" else "not valid"

            else:
                print("---DECISION: MAX RETRIES REACHED---")
                return "max retries"
        
        return "valid"

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
        docs = None
        try:
            docs = self.invoke_with_timeout(self.retriever.invoke, 100, query)
            print("---RELEVANT CHUNKS EXTRACTED---")
        except TimeoutError:
            print("---TIMEOUT OCCURRED---")
            return {
                "timeout": True,
                "generation":"timeout"
            }
        except Exception as e:
            print(f"---AN ERROR OCCURRED: {e}")
        '''
        #Reranker
        doc_texts = [doc.page_content for doc in docs]
        query_doc_pairs = [(query, doc_text) for doc_text in doc_texts]
        scores = self.cross_encoder.predict(query_doc_pairs)
        ranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked_docs[:6]]  
        # Print the top 6 documents
        print("\nTop 5 Documents:")
        for i, doc in enumerate(top_docs, 1):
            print(f"\nDocument {i}:\n{doc.page_content[:1000]}...")  
        #Reranker End
        self.doc_txt = self.format_docs(top_docs)
        '''
        self.doc_txt = self.format_docs(docs)
        rag_prompt_formatted = rag_prompt.format(context=self.doc_txt, question=query)
        try:
            generation = self.invoke_with_timeout(self.custom_llm.invoke, 60, [HumanMessage(content=rag_prompt_formatted)])
        except TimeoutError:
            print("---TIMEOUT OCCURRED---")
            return {
                "timeout": True,
                "generation":"timeout"
            }
        except Exception as e:
            print(f"---AN ERROR OCCURRED: {e}")
        try:
            generation_content = json.loads(generation.content)
            answer = generation_content["answer"]
        except:
            answer = "Error: Answer not in JASON - Format"

        print(f"\nCurrent answer: {answer}")
        return {
            "generation": answer,
            "loop_step": loop_step + 1,
            "timeout": False
        }
     

    def analyze_document(self, query):
        graph = self.workflow.compile()
        # Graph speichern
        #image_data = graph.get_graph().draw_mermaid_png()
        #with open("graph.png", "wb") as f:
            #f.write(image_data)
        inputs = {"question" : query, 
                  "original_question" : query, 
                  "max_retries" : 3}    
        for event in graph.stream(inputs, stream_mode="values"):
            current_state = event
        return current_state['generation']
        
    
        

    

  
    
