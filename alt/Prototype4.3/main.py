from pdf_retriever import PDFRetriever
from llm_access import establish_server_connection, get_llm, get_embedding_model
from mapping import get_chemi_id, get_orpha_code, get_pubmed_metadata, extract_metadata,extract_missing_metadata, get_EudraCT_metadata,get_clinicaltrials_metadata,get_hpos
from validation import validate_db
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json
import sys
import os
import argparse
import time

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
    return {"current_file": None, "restarts" : 0, "output" : {}}

def restart_script(progress):
    progress["restarts"] += 1
    save_progress(STATE_FILE, progress)
    print("\nScript is restarted with current pdf...\n")
    python = sys.executable
    for idx, arg in enumerate(sys.argv):
        if arg == "-p" or arg == "--progression":
            sys.argv[idx + 1] = "True"
    os.execl(python, python, *sys.argv)

def initialize_server(model):
    api_url, headers = establish_server_connection()
    retrieve_llm = get_llm(api_url, headers, model=model)
    mapping_llm = get_llm(api_url, headers)
    embedding_model = get_embedding_model(api_url, headers)
    return retrieve_llm, mapping_llm, embedding_model

def initialize_retriever(file_path, embedding_model, model, keep_references):
    analyzer = PDFRetriever(file_path)
    analyzer.load_pdf(embedding_model, keep_references)
    analyzer.initialize_chain(model)
    return analyzer

def extract_information(analyzer, query, key, output, progress, restart_message):
    
    if key not in output:  
        answer = analyzer.analyze_document(query)
        if answer == "timeout":
            print(restart_message)
            analyzer.delete_db()
            progress["output"] = output
            restart_script(progress)
        output[key] = answer
    return output[key]

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
        ordo_term= list()
        omim=list()
        for diseases in llm_output["disease"]:
            ordo = get_orpha_code(diseases)
            ordo_code.append(ordo["OrphaCode"] if ordo else None)
            ordo_term.append(ordo["Name"] if ordo else None)
            omim.append(ordo["OMIM_Codes"] if ordo else None)
        llm_output["ORDO_code"] = ordo_code if ordo_code else "None"
        llm_output["ORDO_term"] = ordo_term if ordo_code else "None"
        llm_output["OMIM_Codes"] = omim if ordo_code else "None"
    else:
        llm_output["disease"] = ["Timeout occurred"]

    ##ChEBI ID##
    if "treatment" in llm_output:
        chebi = list()
        for treatment in llm_output["treatment"]:
            chebi.append(get_chemi_id(treatment, llm))
        if chebi:
            llm_output["treatment_ID"] = [entry[0] for entry in chebi]
            llm_output["treatment_name_ChEBI"] = [entry[1] for entry in chebi]
    else:
        llm_output["treatment"] = ["Timeout occurred"]

def complete_retrieval(llm_output):
        
    csv_file_path = "test_db.csv"
    try:
        df = pd.read_csv(csv_file_path)
        print("\nCSV file loaded successfully.")
    except FileNotFoundError:
        print("\nCSV file not found. Creating a new file.")
        df = pd.DataFrame(columns=["Publication_database", "Study_identifier",
                                "DOI","Year_of_publication", "Authors", 
                                "Study_title","Journal","disease_LLM","Clinical_diagnosis_ORDO",
                                "ORDO_code","HPO_terms_condition","OMIM_code","gene", "treatment",
                                "treatment_name_ChEBI", "treatment_ID","treatment_effect" ])
        
    new_row = {
        #DATABASE_INFORMATION
        "Publication_database" : llm_output["pub_db"] if "pub_db" in llm_output else "None",
        "Study_identifier" : llm_output["pub_code"],
        "DOI" : llm_output["pub_doi"] if "pub_doi" in llm_output else "None",
        "Year_of_publication" : llm_output["pub_year"] if "pub_year" in llm_output else "None",
        "Authors" : llm_output["pub_authors"] if "pub_authors" in llm_output else "None",
        "Study_title" : llm_output["pub_title"] if "pub_title" in llm_output else "None",
        "Journal" : llm_output["pub_journal"] if "pub_journal" in llm_output else "None",
        #CLINICAL_INFORMATION
        "disease_LLM": llm_output["disease"],
        "Clinical_diagnosis_ORDO":  llm_output["ORDO_term"] if "ORDO_term" in llm_output else "None",
        "ORDO_code": llm_output["ORDO_code"] if "ORDO_code" in llm_output else "None",  
        "HPO_terms_condition": get_hpos(llm_output["ORDO_code"]) if "ORDO_code" in llm_output else "None", 
        #"Molecular_diagnosis_OMIM": "None", #fehlt noch
        "OMIM_code": llm_output["OMIM_Codes"] if "OMIM_Codes" in llm_output else "None",
        
        #GENE-VARIANT_INFORMATION
        "gene": llm_output["gene"] if "gene" in llm_output else "None",  
        #TREATMENT_INFORMATION
        "treatment": llm_output["treatment"],
        #"treatment or intervention database": "ChEBI" if "treatment_ID" in llm_output else "None",
        "treatment_name_ChEBI": llm_output["treatment_name_ChEBI"] if "treatment_name_ChEBI" in llm_output else "None",
        "treatment_ID": llm_output["treatment_ID"] if "treatment_ID" in llm_output else "None",
        "treatment_effect": llm_output["treatment_effect"] if "treatment_effect" in llm_output else "None"
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
    parser.add_argument(
        '-p','--progression', 
        type=str, 
        default="True",
        help="""Can be either set to True or False. 
        True: Continuation with the PDF - document at which the program last stopped.
        False: Starting from the first PDF - document in input directory."""
    )
    parser.add_argument(
        '-r','--references', 
        type=str, 
        default="False",
        help="""Can be either set to True or False. 
        True: The PDF will be processed as it is.
        False: References section will be stripped before analysis for better results.
        Only works if PDF containes exaclty one reference section."""
    )

    args = parser.parse_args()
    file_path = args.file
    model = args.model
    keep_progress = args.progression
    keep_references = args.references

    initial_query = """What is the single primary disease addressed in the paper?
    Keep the answer concise and limited to the disease name with its type or subtype, 
    written in full, without any additional details, descriptions, or classifications."""

    follow_up_querys = [
           """The paper addresses the following disease: {answer}.  
            What are the primary treatment option(s) explicitly associated with this disease, as suggested in the paper?
            Ensure the treatments are stated with precision, avoiding vague terms such as "dietary supplementation" or "pharmaceutical therapy."  
            If the treatment involves medications or substances, provide only the name of the substance, with no additional details regarding dosage,
            administration, or qualifiers such as "intermediate dose" or "therapy." Include only substances for which there is a reported outcome,
            whether positive or negative. Do not mention placebos under any circumstances.  
            List only the names of the treatments without any additional information or descriptions. """,

            """The paper discusses the following disease: {answer}.
            Identify the gene(s) that are directly and specifically associated with this disease according to the paper.
            Exclude any genes mentioned in introductory or background sections, or those that are only loosely related to the disease.
            Only extract genes that are central and significant in the main context of the paper's discussion about the disease.
            Provide only the names of the gene(s), such as: IGF1, TNFA, PPARG, NLGN1. Don’t add any further information""",
            """The Paper addresses the following treatment(s): {treatment}. 
            Evaluate each treatment option's effect and classify it into one of the following categories: 
            n/a, None, small, medium, or large. Return the classification for each treatment separately."""
            ]

    retrieve_llm, mapping_llm, embedding_model = initialize_server(model)
    file_list = os.listdir(file_path)
    if keep_progress != "True":
        progress = {"current_file": None, "restarts" : 0, "output" : {}}
    else:
        progress = load_progress(STATE_FILE)
        if progress["current_file"] and progress["current_file"] not in file_list:
            progress = {"current_file": None, "restarts" : 0, "output" : {}}

    start_index = file_list.index(progress["current_file"]) + 1 if progress["current_file"] else 0

    if progress["restarts"] == MAX_RESTARTS:
        print("---MAXIMUM RESTARTS REACHED---")
        output = progress["output"]
        analyzer = initialize_retriever(
            file_path +"/"+ file_list[start_index], 
            embedding_model, 
            retrieve_llm, 
            keep_references
        )
        map_metadata(output, file_path +"/"+ file_list[start_index], analyzer, mapping_llm)
        analyzer.delete_db()
        complete_retrieval(output)
        print("move on to the next pdf...")
        progress["restarts"] = 0
        progress["output"] = {}
        start_index += 1 
        if start_index < len(file_list):
            progress["current_file"] = file_list[start_index]

    for file in file_list[start_index:]:
        # Initialize output and analyzer
        output = progress.get("output", {})
        analyzer = initialize_retriever(
            file_path + "/" + file, 
            embedding_model, 
            retrieve_llm,
            keep_references
            )
        
        # Extract "disease"
        disease = extract_information(
            analyzer,
            query=initial_query,
            key="disease",
            output=output,
            progress=progress,
            restart_message="---DISEASE INFORMATION COULD NOT BE EXTRACTED---"
        )
        
        # Konvert "disease" in a list format (if necessary)
        if isinstance(disease, list):
            disease_str = ", ".join(disease)
        else:
            disease_str = disease
            disease = [disease]
        
        output["disease"] = disease
        
        queries = {
            "treatment": follow_up_querys[0].format(answer=disease_str),
            "gene": follow_up_querys[1].format(answer=disease_str),
        }
    
        print("\n---Starting parallel extraction of treatment and gene information---".upper())
    
        def parallel_extract(key):
            return extract_information(
                analyzer,
                query=queries[key],
                key=key,
                output=output,
                progress=progress,
                restart_message=f"---{key.upper()} INFORMATION COULD NOT BE EXTRACTED---"
            )
    
        with ThreadPoolExecutor() as executor:
            # Map keys ("treatment", "gene") to the parallel_extract function
            results = executor.map(parallel_extract, queries.keys())
    
        # Store the results in the output dictionary
        for key, result in zip(queries.keys(), results):
            output[key] = result


        #Clinical effect
        if "treatment" in output and output["treatment"]:
            treatments = output["treatment"]
            treatments_str = ", ".join(treatments)

            effect_query = follow_up_querys[2].format(treatment=treatments_str)
            print("\n---Starting extraction of treatment effects---".upper())

    
            treatment_effects = extract_information(
                analyzer,
                query=effect_query,
                key="treatment_effect",
                output=output,
                progress=progress,
                restart_message="---TREATMENT EFFECT INFORMATION COULD NOT BE EXTRACTED---"
            )

            output["treatment_effect"] = treatment_effects
        else:
            print("---NO TREATMENTS FOUND, SKIPPING EFFECT EXTRACTION---")
            output["treatment_effect"] = "None"
        #Clinical Effect End
        
        print(f"\n{model} final response: {output}")
        print("\nExtracting metadata...")
        map_metadata(output, file_path +"/"+ file, analyzer, mapping_llm)
        analyzer.delete_db()
        complete_retrieval(output)
        progress["current_file"] = file
        progress["restarts"] = 0
        progress["output"] = {}
        save_progress(STATE_FILE, progress)  # Save progress

    #Reset Progress File
    progress["current_file"] = None
    progress["restarts"] = 0
    progress["output"] = {}
    save_progress(STATE_FILE, progress)

def validate():
    test_data = pd.read_csv("test_db.csv")
    vali_data = pd.read_csv("validation_data.csv", sep=";")
    test_data['Study_identifier'] = test_data['Study_identifier'].astype(str)
    vali_data['Study_identifier'] = vali_data['Study_identifier'].astype(str)

    validate_db(test_data, vali_data)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Die Laufzeit des Codes beträgt {elapsed_time:.4f} Sekunden.")
    #validate()
