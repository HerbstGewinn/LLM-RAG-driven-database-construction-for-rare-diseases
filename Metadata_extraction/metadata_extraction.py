from Bio import Entrez, Medline
import re, os
from pdfminer.high_level import extract_text

def get_EudraCT_metadata(file_path):
    text = extract_text(file_path)
    code_match = re.search(r"EudraCT Number:\s*(\S+)", text)
    if not code_match:
        code_match = re.search(r"EudraCT number\s*-\s*(\S+-\s*\S+) \(\d{4}\)", os.path.basename(file_path))
    year_match = re.search(r"\((\d{4})\)", os.path.basename(file_path))
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
    pub_code = re.search(r".*/(.+)\.pdf", file_path).group(1)
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

def extract_missing_metadata():
    # LLM should search in document for the missing parameter(s) (all that are "N/A")

def map_metadata(llm_output, file_path, analyzer, llm):
        
    ##Study metadata##
    if "EudraCT" in os.path.basename(file_path):
        study_data = get_EudraCT_metadata(file_path)
    else:
        study_data = get_pubmed_metadata(file_path)
        if not study_data:
            study_data = get_clinicaltrials_metadata(file_path)
            if not study_data: # none of the three publication databases
                instructions = """
                    You are an information extractor.
                    Find meta information about the DOCUMENT. 
                    Focus on:\n\n-code (pub_code)\n-doi (pub_doi)\n-year of publication (pub_year)
                    -author names (pub_authors)\n-title of the document (pub_title)
                    -journal (pub_journal)\n"""
                docs = analyzer.retriever.invoke(instructions)
                docs_txt = analyzer.format_docs(docs)
                study_data = extract_metadata(docs_txt, llm, instructions)
    # for all the parameters that are "N/A", extract_missing_metadata
    for key in study_data:
        llm_output[key] = study_data[key]

    print("---EXTRACTED STUDY METADATA---")
    #...