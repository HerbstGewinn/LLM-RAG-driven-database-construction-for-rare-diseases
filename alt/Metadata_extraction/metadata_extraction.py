from Bio import Entrez, Medline
import re, os
from pdfminer.high_level import extract_text

# functions in mapping.py:

def extract_metadata(doc, llm, instructions):
    prompt = f"""
        DOCUMENT: 
        {doc}

        You MUST strictly follow this JSON format:
        {{
            "pub_code": "string"
            "pub_doi": "string",
            "pub_year": "string",
            "pub_authors": ["string"],
            "pub_title" : "string",
            "pub_journal" : "string"
        }}
        """
    
    answer = llm.invoke(
            [SystemMessage(content=instructions)] + [HumanMessage(content=prompt)]
        )
    answer_content = json.loads(answer.content)
    return answer_content

def extract_missing_metadata(doc, llm, instructions):
    prompt = f"""
        DOCUMENT:
        {doc}

        You MUST strictly follow this format: "string"
        """
    answer = llm.invoke(
        [SystemMessage(content=instructions)] + [HumanMessage(content=prompt)])
    answer_content = json.loads(answer.content)
    return answer_content

def get_EudraCT_metadata(file_path):
    text = extract_text(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    code_match = re.search(r"EudraCT Number:\s*(\S+)", text)
    if not code_match:
        code_match = re.search(r"EudraCT number\s*-\s*(\S+-\s*\S+) \(\d{4}\)", file_name)
    year_match = re.search(r"\((\d{4})\)", file_name)
    authors_match = re.search(r"Name of Sponsor:\s*(.+)", text)
    metadata = {
        "pub_db": "EU Clinical Trials",
        "pub_code": code_match.group(1).replace(" ","") if code_match else "N/A",
        "pub_doi": "",
        "pub_year": year_match.group(1) if year_match else "N/A",
        "pub_authors": authors_match.group(1).strip() if authors_match else "N/A",
        "pub_title": "N/A",
        "pub_journal": ""
    }
    return metadata

def get_clinicaltrials_metadata(file_path):
    text = extract_text(file_path)
    code_match = re.search(r"ClinicalTrials\.gov ID\s*(\S+)", text)
    if not code_match:
        return None
    author_match = re.search(r"Sponsor\s*(.+)", text)
    metadata = {
        "pub_db": "ClinicalTrials.org",
        "pub_code": code_match.group(1) if code_match else "N/A",
        "pub_doi": "",
        "pub_year": "N/A",
        "pub_authors": author_match.group(1) if author_match else "N/A",
        "pub_title": "N/A",
        "pub_journal": ""
    }
    return metadata

def get_pubmed_metadata(file_path, email=""):
    Entrez.email = email
    pub_code = os.path.splitext(os.path.basename(file_path))[0]
    search_handle = Entrez.esearch(db="pubmed", term=pub_code, retmax=1) # Only top result
    search_results = Entrez.read(search_handle)
    search_handle.close()
    if not search_results["IdList"]:
        return None
    
    # Fetch metadata for the top result
    article_id = search_results["IdList"][0]
    if article_id != pub_code:
        return None
    stream = Entrez.efetch(db="pubmed", id=article_id, rettype="medline", retmode="text")
    record = Medline.read(stream)
    stream.close()

    elocation_ids = record["AID"]
    doi = "N/A"
    for elocation in elocation_ids:
        if '[doi]' in elocation:
            doi = elocation.split(' [doi]')[0]
            break

    metadata = {
        "pub_db": "Pubmed",
        "pub_code": record["PMID"] if record["PMID"] else "N/A",
        "pub_doi": doi,
        "pub_year": record["DP"].split()[0] if record["DP"] else "N/A",
        "pub_authors": record["FAU"] if record["FAU"] else "N/A",
        "pub_title": record["TI"] if record["TI"] else "N/A",
        "pub_journal": record["JT"] if record["JT"] else "N/A",
        #"pub_abstract": record["AB"] if record["AB"] else "N/A"
    }
    return metadata




# functions in main.py:

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
            docs_txt = analyzer.format_docs(docs)
            study_data[key] = extract_missing_metadata(docs_txt, llm, instructions)
        llm_output[key] = study_data[key]

    print("---EXTRACTED STUDY METADATA---")
    #...

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
        "DOI" : llm_output["pub_doi"],
        "Year_of_publication" : llm_output["pub_year"],
        "Authors" : llm_output["pub_authors"],
        "Study_title" : llm_output["pub_title"],
        "disease": llm_output["disease"],
        "treatment": llm_output["treatment"],
        "gene": llm_output["gene"],  
        "ORDO_code": llm_output["ORDO_code"],
        "treatment_ID": llm_output["treatment_ID"]
        }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_file_path, index=False)
    print("\n---RETRIEVAL SAVED SUCCESSFULLY---\n")