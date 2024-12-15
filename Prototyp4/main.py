from pdf_retriever import PDFRetriever
from mapping import get_chemi_id, get_orpha_code
from validation import validate_db
import pandas as pd
import re
import sys
import os
#from langchain.agents import Tool

sys.stdout.reconfigure(encoding='utf-8')

def initialize(file_path, model):
    analyzer = PDFRetriever(file_path, model)
    analyzer.load_pdf()
    analyzer.initialize_chain()
    return analyzer

def complete_retrieval(llm_output, file_path):
        
    csv_file_path = "test_db.csv"
    try:
        df = pd.read_csv(csv_file_path)
        print("\nCSV file loaded successfully.")
    except FileNotFoundError:
        print("\nCSV file not found. Creating a new file.")
        df = pd.DataFrame(columns=["Study_identifier","disease",
                                "treatment","gene","ORDO_code","treatment_ID"])
        
    study_id = re.search(r".*/(.+)\.pdf", file_path)

    ordo_code = list()
    for diseases in llm_output["disease"]:
        ordo_code.append(get_orpha_code(diseases)["OrphaCode"])

    chemi_id = list()
    for treatment in llm_output["treatment"]:
       chemi_id.append(get_chemi_id(treatment))

    new_row = {
        "Study_identifier": study_id.group(1) if study_id else "None",
        "disease": llm_output["disease"],
        "treatment": llm_output["treatment"],
        "gene": llm_output["gene"],  
        "ORDO_code": ordo_code if ordo_code else "None",
        "treatment_ID": chemi_id if chemi_id else "None"
        }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_file_path, index=False)
    print("\nRetrieval saved successfully.")


file_path = "C:/Users/Adrian/Desktop/Bioinformatik/Projekt/Treatbolome/test"
model = "qwen2.5:72b"
#model = "meditron:70b"

initial_query = """What primary disease does the paper address? 
Classify it by type or subtype, if applicable. Exactly one answer is required, not several"""
follow_up_querys = [
          """The Paper addresses the following disease: {answer}. 
          What primary treatment option(s) specifically associated with this disease 
          are suggested in the paper? Ensure that the treatments are described with precision, 
          avoiding vague terms like 'dietary supplementation' or 'pharmaceutical therapy'. 
          Include a maximum of two treatment options. 
          There is no need for additional information, besides the treatment option(s)""",
          """The Paper addresses the following disease: {answer}. 
          What gene(s) are directly associated with this disease according to the paper?
          Ensure that only the gene(s) are extracted. 
          There is no need for additional information, besides the gene(s)."""
          ]
back_up_query = ""

for file in os.listdir(file_path):
    output = dict()
    analyzer = initialize(file_path +"/"+ file, model)
    answer = analyzer.analyze_document(initial_query)
    if isinstance(answer, list):
        disease = ", ".join(item for item in answer)
    else:
        disease = answer
        answer = [answer]
    output["disease"] = answer
    treatment = analyzer.analyze_document(follow_up_querys[0].format(answer=disease))
    output["treatment"] = treatment
    gene = analyzer.analyze_document(follow_up_querys[1].format(answer=disease))
    output["gene"] = gene
    analyzer.delete_db()
    print(f"\n{model} final response: {output}")
    complete_retrieval(output, file_path +"/"+ file)

test_data = pd.read_csv("test_db.csv")
vali_data = pd.read_csv("validation_data.csv", sep=";")
test_data['Study_identifier'] = test_data['Study_identifier'].astype(str)
vali_data['Study_identifier'] = vali_data['Study_identifier'].astype(str)

validate_db(test_data, vali_data)

