from pdf_retriever import PDFRetriever
from llm_access import establish_server_connection, get_llm, get_embedding_model
from mapping import get_chemi_id, get_orpha_code, get_pubmed_metadata, extract_metadata
from validation import validate_db
import pandas as pd
import re
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

def initialize(file_path, embedding_model, model):
    analyzer = PDFRetriever(file_path)
    analyzer.load_pdf(embedding_model)
    analyzer.initialize_chain(model)
    return analyzer

def map_metadata(llm_output, file_path, llm):
        
    ##Study metadata##
    study_id = re.search(r".*/(.+)\.pdf", file_path)
    if study_id:
        _id = study_id.group(1)
        llm_output["Study_identifier"] = f"{_id}"
        study_data = get_pubmed_metadata(_id)
    else: 
        llm_output["Study_identifier"] = "None"
        study_data = None
    if not study_data:
        instructions = """
            You are an information extractor.
            Find meta information about the DOCUMENT. 
            Focus on:\n\n-doi (pub_doi)\n-year of publication (pub_year)
            -author names (pub_authors)\n-title of the document (pub_title)
            -journal (pub_journal)\n"""
        docs = analyzer.retriever.invoke(instructions)
        docs_txt = analyzer.format_docs(docs)
        study_data = extract_metadata(docs_txt, llm, instructions)
        
    for key in study_data:
        llm_output[key] = study_data[key]

    print("---EXTRACTED STUDY METADATA---")

    ##ORDO Code##
    ordo_code = list()
    for diseases in llm_output["disease"]:
        ordo_code.append(get_orpha_code(diseases)["OrphaCode"])
    llm_output["ORDO_code"] = ordo_code if ordo_code else "None"

    ##ChEBI ID##
    chemi_id = list()
    for treatment in llm_output["treatment"]:
       chemi_id.append(get_chemi_id(treatment, llm))
    llm_output["treatment_ID"] = chemi_id if chemi_id else "None"

def complete_retrieval(llm_output):
        
    csv_file_path = "test_db.csv"
    try:
        df = pd.read_csv(csv_file_path)
        print("\nCSV file loaded successfully.")
    except FileNotFoundError:
        print("\nCSV file not found. Creating a new file.")
        df = pd.DataFrame(columns=["Puplication_database", "Study_identifier",
                                "DOI","Year_of_publication", "Authors", 
                                "Study_title", "disease", "treatment","gene",
                                "ORDO_code","treatment_ID"])
        
    new_row = {
        "Puplication_database" : llm_output["pub_db"] if "pub_db" in llm_output else "None",
        "Study_identifier" : llm_output["Study_identifier"],
        "DOI" : llm_output["pub_doi"] if "pub_doi" in llm_output else "None",
        "Year_of_publication" : llm_output["pub_year"] if "pub_year" in llm_output else "None",
        "Authors" : llm_output["pub_authors"] if "pub_authors" in llm_output else "None",
        "Study_title" : llm_output["pub_title"] if "pub_title" in llm_output else "None",
        "disease": llm_output["disease"],
        "treatment": llm_output["treatment"],
        "gene": llm_output["gene"],  
        "ORDO_code": llm_output["ORDO_code"],
        "treatment_ID": llm_output["treatment_ID"]
        }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_file_path, index=False)
    print("\n---RETRIEVAL SAVED SUCCESSFULLY---\n")


file_path = "C:/Users/Adrian/Desktop/Bioinformatik/Projekt/Treatbolome/test1"

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

model="qwen2.5:72b"

api_url, headers = establish_server_connection()
retrieve_llm = get_llm(api_url, headers, model=model)
mapping_llm = get_llm(api_url, headers)
embedding_model = get_embedding_model(api_url, headers)

for file in os.listdir(file_path):
    output = dict()
    analyzer = initialize(file_path +"/"+ file, embedding_model, retrieve_llm)
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
    print(f"\n{model} final response: {output}")
    print("\nExtracting metadata...")
    map_metadata(output, file_path +"/"+ file, mapping_llm)
    analyzer.delete_db()
    complete_retrieval(output)

test_data = pd.read_csv("test_db.csv")
vali_data = pd.read_csv("validation_data.csv", sep=";")
test_data['Study_identifier'] = test_data['Study_identifier'].astype(str)
vali_data['Study_identifier'] = vali_data['Study_identifier'].astype(str)

validate_db(test_data, vali_data)

