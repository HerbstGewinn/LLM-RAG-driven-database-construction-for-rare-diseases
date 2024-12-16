from fuzzy_orpha_mapper import Fuzzy_Orpha_Mapper
from bioservices import ChEBI
import requests
from langchain_core.messages import HumanMessage, SystemMessage
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def get_chemi_id(treatment, llm):
    instructions = """
    
    You are an information extractor tasked with **extracting exactly one chemical substance** from a given treatment string. 

    Instructions:
    1. If the string contains a substance, extract **exactly one** and return it as the result. If multiple substances are mentioned, choose the most relevant one based on the context.
    2. If the string does not contain any substances, return 'None'.
    3. Pay attention to the context. For example, if the text suggests avoiding substances or mentions "as little substance as possible," it is not a treatment.

    Examples:

    INPUT STRING: 'high-dose oral riboﬂavin therapy', EXTRACTED SUBSTANCE: 'riboﬂavin'.
    INPUT STRING: 'Ascorbic acid', EXTRACTED SUBSTANCE: 'Ascorbic acid'. 
    INPUT STRING: 'Resistance training exercise', EXTRACTED SUBSTANCE: 'None'.
    INPUT STRING: 'A diet low in phytanic acid', EXTRACTED SUBSTANCE: 'None'.

    You MUST strictly follow this JSON format:
    {{"substance" : "string"}}
    """
    answer = llm.invoke(
            [SystemMessage(content=instructions)] + [HumanMessage(content=treatment)]
        )
    answer_content = json.loads(answer.content)
    print(f"---EXTRACTED TREATMENT SUBSTANCE: {answer_content["substance"]}---")
    if answer_content["substance"] == "None": 
        return None
    
    chebi = ChEBI()

    result = chebi.getLiteEntity(answer_content["substance"], searchCategory="CHEBI NAME", maximumResults=5)
    
    if result:
        top_score = result[0]["searchScore"]
        best_results = [entity['chebiAsciiName'] for entity in result if entity["searchScore"] == top_score] #If there are several results with the same score
        if len(best_results) == 1:
            entity = result[0]
            print(f"Best match: ChEBI ID: {entity['chebiId']}, Name: {entity['chebiAsciiName']}, Score: {entity["searchScore"]}")
            return entity['chebiId']
        else:
            search_instructions = """
            You are an information extractor tasked with finding the best matching substance in a LIST of substances.
            You are given a LIST of substances and a SUBSTANCE. 
            Return the substance in the LIST that best matches the given SUBSTANCE. 
            
            You MUST strictly follow this JSON format:
            {{"substance" : "string"}}"""

            query = f"SUBSTANCE: {answer_content["substance"]}\n\nLIST: {best_results}"

            best_result = llm.invoke(
                [SystemMessage(content=search_instructions)] + [HumanMessage(content=query)]
            )
            best_result_content = json.loads(best_result.content)
            for idx, name in enumerate(best_results):
                if name == best_result_content["substance"]:
                    print(f"Best match: ChEBI ID: {result[idx]['chebiId']}, Name: {result[idx]['chebiAsciiName']}, Score: {result[idx]["searchScore"]}")
                    return result[idx]['chebiId']
            return result[0]['chebiId']


    else:
        print("No results found.")
        return None

def get_orpha_code(disease):
    disease_processor = Fuzzy_Orpha_Mapper('en_product1.json')

    orpha_code = disease_processor.find_orpha_code(disease)

    return orpha_code if orpha_code else "None"

if __name__ =="__main__":
    treatments = ['Resistance training exercise', 'Creatine supplementation',
                'strict adherence to the Chelsea diet', 
                'Oral curcumin at 50mg/kg/day for the first 4 months, then 75mg/kg/day for the remaining 8 months.',
                'Ascorbic acid', 'A diet low in phytanic acid', 'Extracorporeal lipid apheresis', 'Tafamidis (20 mg QD)', 'Liver transplantation',
                'carbidopa', 'Ascorbic acid (4 g/d)', 'high-dose oral riboﬂavin therapy', 'diflunisal 250 mg twice daily', 'Tafamidis / Fx-1006A',
                'L-serine supplementation'] 



