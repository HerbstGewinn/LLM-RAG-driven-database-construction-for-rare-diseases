import operator
from typing_extensions import TypedDict
from typing import List, Annotated

class GraphState(TypedDict):

    question : str #User question
    original_question : str #Original question in case query gets modified
    generation : list[str] #LLM generation 
    max_retries : int #Max number of retries for answer generation 
    loop_step : Annotated[int, operator.add]
    is_hallucinating : str #Status of current hallucination check
    is_valid : str #Status of answer validation check 
    current_explanation : str #Current explanation for lack of quality