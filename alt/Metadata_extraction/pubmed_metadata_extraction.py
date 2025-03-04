from Bio import Entrez, Medline

def fetch_pubmed_metadata(pub_code, email): #api_key):
    Entrez.email = email
    #Entrez.api_key = api_key
    search_handle = Entrez.esearch(db="pubmed", term=pub_code, retmax=1) # Only top result
    search_results = Entrez.read(search_handle)
    search_handle.close()
    if not search_results["IdList"]:
        return "Error: No results found on PubMed."
    
    # Fetch metadata for the top result
    article_id = search_results["IdList"][0]
    #if article_id != pub_code:
    #    return "Error: No results found on PubMed."
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
        "pub_code": record["PMID"],
        "pub_doi": doi,
        "pub_year": record["DP"].split()[0], # only year from date
        "pub_authors": record["FAU"],
        "pub_title": record["TI"],
        "pub_journal": record["JT"],
        #"pub_abstract": record["AB"]
    }
    return metadata

pubmed_id = "29042446" # Also works with title

email = "" # Always tell NCBI who you are!
#api_key = ""

metadata = fetch_pubmed_metadata(pubmed_id, email) #api_key)

print(metadata)