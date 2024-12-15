from bioservices import ChEBI
from rapidfuzz import fuzz
import requests
from fuzzy_orpha_mapper import Fuzzy_Orpha_Mapper

def get_chemi_id(query):
    chebi = ChEBI()

    results = chebi.getLiteEntity(query, searchCategory="CHEBI NAME", maximumResults=10)

    if results:
        best_match = None
        best_score = 0

        for entity in results:  
            chebi_name = entity['chebiAsciiName']
            score = fuzz.ratio(query.lower(), chebi_name.lower())  
            if score > best_score:  
                best_score = score
                best_match = entity

        if best_match:
            print(f"Best match: ChEBI ID: {best_match['chebiId']}, Name: {best_match['chebiAsciiName']}, Score: {best_score}")
            return best_match['chebiId']
        else:
            print("No fitting match found.")
            return results[0]['chebiId']
    else:
        print("No results found.")
        return None

def get_orpha_code(disease):
    disease_processor = Fuzzy_Orpha_Mapper('en_product1.json')

    orpha_code = disease_processor.find_orpha_code(disease)

    return orpha_code if orpha_code else "None"

