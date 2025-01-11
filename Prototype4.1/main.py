from pdf_retriever import PDFRetriever
from llm_access import establish_server_connection, get_llm, get_embedding_model
from mapping import get_chemi_id, get_orpha_code, get_pubmed_metadata, extract_metadata,extract_missing_metadata, get_EudraCT_metadata,get_clinicaltrials_metadata
from validation import validate_db
import pandas as pd
import json
import sys
import os
import argparse

sys.stdout.reconfigure(encoding='utf-8')
MAX_RESTARTS = 3
STATE_FILE = "progress.json"

def save_progress(file_name, progress):
    with open(file_name, "w") as file:
        json.dump(progress, file)

def load_progress(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r") as file:
            return json.load(file)
    return {"current_file": None, "restarts" : 0}

def restart_script(progress):
    progress["restarts"] += 1
    save_progress(STATE_FILE, progress)
    print("\nScript is restarted with current pdf...\n")
    python = sys.executable
    os.execl(python, python, *sys.argv)

def initialize_server(model):
    api_url, headers = establish_server_connection()
    retrieve_llm = get_llm(api_url, headers, model=model)
    mapping_llm = get_llm(api_url, headers)
    embedding_model = get_embedding_model(api_url, headers)
    return retrieve_llm, mapping_llm, embedding_model

def initialize_retriever(file_path, embedding_model, model):
    analyzer = PDFRetriever(file_path)
    analyzer.load_pdf(embedding_model)
    analyzer.initialize_chain(model)
    return analyzer

def map_metadata(llm_output, file_path, analyzer, llm):
        
    ##Study metadata##
    if "EudraCT" in os.path.splitext(os.path.basename(file_path))[0]:
        study_data = get_EudraCT_metadata(file_path)
    else:
        study_data = get_pubmed_metadata(file_path)
        if not study_data:
            study_data = get_clinicaltrials_metadata(file_path)
            if not study_data: # none of the three publication databases
                instructions = """
                    You are an information extractor.
                    Find meta information about the DOCUMENT. 
                    Focus on:\n\n-study identifier (pub_code)\n-doi (pub_doi)
                    -year of publication (pub_year)\n-author names (pub_authors)
                    -title of the document (pub_title)\n-journal (pub_journal)\n"""
                docs = analyzer.retriever.invoke(instructions)
                if docs == "timeout":
                    study_data = {"pub_db" : "timeout"}
                else:
                    docs_txt = analyzer.format_docs(docs)
                    study_data = extract_metadata(docs_txt, llm, instructions)
    for key in study_data:
        if study_data[key] == "N/A":
            instructions = f"""
                You are an information extractor.
                Find out only the {key} of the DOCUMENT.
                (pub_code = study identifier, pub_year = year of publication,
                pub_authors = author names, pub_title = title of the document,
                pub_journal = journal)
                """
            docs = analyzer.retriever.invoke(instructions)
            if docs == "timeout":
                study_data[key] = "timeout"
            else:
                docs_txt = analyzer.format_docs(docs)
                study_data = extract_metadata(docs_txt, llm, instructions)
        llm_output[key] = study_data[key]

    print("---EXTRACTED STUDY METADATA---")
    print("Mapping extracted inforation...")

    ##ORDO Code##
    if "disease" in llm_output:
        ordo_code = list()
        for diseases in llm_output["disease"]:
            ordo = get_orpha_code(diseases)
            ordo_code.append(ordo["OrphaCode"] if ordo != "None" else "None")
        llm_output["ORDO_code"] = ordo_code if ordo_code else "None"
    else:
        llm_output["disease"] = ["Timeout occurred"]

    ##ChEBI ID##
    if "treatment" in llm_output:
        chemi_id = list()
        for treatment in llm_output["treatment"]:
            chemi_id.append(get_chemi_id(treatment, llm))
        llm_output["treatment_ID"] = chemi_id if chemi_id else "None"
    else:
        llm_output["treatment"] = ["Timeout occurred"]

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
        "Study_identifier" : llm_output["pub_code"],
        "DOI" : llm_output["pub_doi"] if "pub_doi" in llm_output else "None",
        "Year_of_publication" : llm_output["pub_year"] if "pub_year" in llm_output else "None",
        "Authors" : llm_output["pub_authors"] if "pub_authors" in llm_output else "None",
        "Study_title" : llm_output["pub_title"] if "pub_title" in llm_output else "None",
        "disease": llm_output["disease"],
        "treatment": llm_output["treatment"],
        "gene": llm_output["gene"] if "gene" in llm_output else "None",  
        "ORDO_code": llm_output["ORDO_code"] if "ORDO_code" in llm_output else "None",
        "treatment_ID": llm_output["treatment_ID"] if "treatment_ID" in llm_output else "None"
        }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_file_path, index=False)
    print("\n---RETRIEVAL SAVED SUCCESSFULLY---\n")

def main():
    parser = argparse.ArgumentParser(description="Program for document (pdf) analysis with different LLMs.")
    # Choose model
    parser.add_argument(
        '-m','--model', 
        type=str, 
        default="qwen2.5:72b",
        help="Which model do you want to use? (Standard: qwen2.5:72b)"
    )
    # Choose file
    parser.add_argument(
        '-f','--file', 
        type=str, 
        required=True,
        help="Path to PDF - directory. Example: C:/Users/Desktop/PDF_folder"
    )

    args = parser.parse_args()
    file_path = args.file
    model = args.model

    initial_query = """What is the single primary disease addressed in the paper?
    Keep the answer concise and limited to the disease name with its type or subtype, written in full, without any additional details, descriptions, or classifications."""

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

    retrieve_llm, mapping_llm, embedding_model = initialize_server(model)
    progress = load_progress(STATE_FILE)
    start_index = os.listdir(file_path).index(progress["current_file"]) + 1 if progress["current_file"] else 0

    if progress["restarts"] == MAX_RESTARTS:
        print("---MAXIMUM RESTARTS REACHED---")
        output = dict()
        analyzer = initialize_retriever(file_path +"/"+ os.listdir(file_path)[start_index], embedding_model, retrieve_llm)
        map_metadata(output, file_path +"/"+ os.listdir(file_path)[start_index], analyzer, mapping_llm)
        analyzer.delete_db()
        complete_retrieval(output)
        print("move on to the next pdf...")
        progress["restarts"] = 0
        start_index += 1 
        if start_index < len(os.listdir(file_path)):
            progress["current_file"] = os.listdir(file_path)[start_index]

    for file in os.listdir(file_path)[start_index:]:
        output = dict()
        analyzer = initialize_retriever(file_path +"/"+ file, embedding_model, retrieve_llm) #After Server Timeout it must be reinitialized  
        answer = analyzer.analyze_document(initial_query)
        if answer == "timeout": #Answer for disease is mandatory to proceed
            print("---DISEASE INFORMATION COULD NOT BE EXTRACTED---")
            analyzer.delete_db()
            restart_script(progress)
       
        if isinstance(answer, list):
            disease = ", ".join(item for item in answer)
        else:
            disease = answer
            answer = [answer]
            
        output["disease"] = answer
        treatment = analyzer.analyze_document(follow_up_querys[0].format(answer=disease))
        if treatment == "timeout":
            print("---TREATMENT INFORMATION COULD NOT BE EXTRACTED---")
            analyzer.delete_db()
            restart_script(progress)
        
        output["treatment"] = treatment
        gene = analyzer.analyze_document(follow_up_querys[1].format(answer=disease))
        if gene == "timeout":
            print("---TREATMENT INFORMATION COULD NOT BE EXTRACTED---")
            analyzer.delete_db()
            restart_script(progress)

        output["gene"] = gene
        print(f"\n{model} final response: {output}")
        print("\nExtracting metadata...")
        map_metadata(output, file_path +"/"+ file, analyzer, mapping_llm)
        analyzer.delete_db()
        complete_retrieval(output)
        progress["current_file"] = file
        progress["restarts"] = 0
        save_progress(STATE_FILE, progress)  # Fortschritt speichern

    #Reset Progress File
    progress["current_file"] = None
    progress["restarts"] = 0
    save_progress(STATE_FILE, progress)

    test_data = pd.read_csv("test_db.csv")
    vali_data = pd.read_csv("validation_data.csv", sep=";")
    test_data['Study_identifier'] = test_data['Study_identifier'].astype(str)
    vali_data['Study_identifier'] = vali_data['Study_identifier'].astype(str)

    validate_db(test_data, vali_data)

if __name__ == "__main__":
    main()
