from bert_orpha_omim_mapper import BertOrphaOmimMapper
from bioservices import ChEBI
from Bio import Entrez, Medline
from langchain_core.messages import HumanMessage, SystemMessage
import json
import sys
import re, os
from pdfminer.high_level import extract_text

sys.stdout.reconfigure(encoding='utf-8')

def extract_metadata(doc, llm, instructions, file_path):
    file_name = os.path.basename(file_path)  
    file_name = os.path.splitext(file_name)[0]  
    
    prompt = f"""
        DOCUMENT: 
        {doc}

        You MUST strictly follow this JSON format:
        {{
            "pub_code": "{file_name}",
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
    try:
        answer_content = json.loads(answer.content)
        answer_content["pub_code"] = file_name  
        return answer_content
    except:
        return {}
    
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
    search_handle = Entrez.esearch(db="pubmed", term=pub_code, retmax=1) 
    search_results = Entrez.read(search_handle)
    search_handle.close()
    if not search_results["IdList"]:
        return None
    
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
    try:
        answer = llm.invoke(
                [SystemMessage(content=instructions)] + [HumanMessage(content=treatment)]
            )
        
        answer_content = json.loads(answer.content)
        print(f"---EXTRACTED TREATMENT SUBSTANCE: {answer_content["substance"]}---")
        if answer_content["substance"] == "None": 
            return None, None
    except:
        answer_content = {"substance" : treatment}
    
    chebi = ChEBI()

    result = chebi.getLiteEntity(answer_content["substance"], searchCategory="CHEBI NAME", maximumResults=5)
    
    if result:
        top_score = result[0]["searchScore"]
        best_results = [entity['chebiAsciiName'] for entity in result if entity["searchScore"] == top_score] 
        if len(best_results) == 1:
            entity = result[0]
            print(f"Best match: ChEBI ID: {entity['chebiId']}, Name: {entity['chebiAsciiName']}, Score: {entity["searchScore"]}")
            return entity['chebiId'], entity['chebiAsciiName']
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
                    return result[idx]['chebiId'], result[idx]['chebiAsciiName']
            return result[0]['chebiId'], result[0]['chebiAsciiName']

    else:
        return None, None

def get_orpha_omim(disease):
    disease_processor = BertOrphaOmimMapper("en_product1.json")  

    orpha_omim = disease_processor.find_orpha_omim(disease)

    return orpha_omim if orpha_omim else None

def create_hpo_dict(): 
  input_file=("phenotype.hpoa")
  hpo_dict = dict()
  with open(input_file,'r') as infile:
    for _ in range(5):
      next(infile)
    for line in infile:
      columns = line.strip().split('\t')
      code = columns[0]
      hpos = columns[3]
      if code not in hpo_dict:
        hpo_dict[code] = list()
      hpo_dict[code].append(hpos)
  return hpo_dict

def get_hpos(code): 
  code_int=int(code[0])
  hpo_dict = create_hpo_dict()
  code_int = f"{"ORPHA"}:{code_int}"
  hpos = hpo_dict.get(code_int)
  return hpos




