from pdf_retriever import PDFRetriever
from mapping import get_chemi_id, get_orpha_code
import pandas as pd
import os
import re
import sys
from tabulate import tabulate

sys.stdout.reconfigure(encoding='utf-8')
 
def analyze_pdf(file_path, model, query): 
    analyzer = PDFRetriever(file_path, model)
    analyzer.load_pdf()
    analyzer.initialize_chain()

    response = analyzer.analyze_document(query)
    print(f"\n{model} response:")
    print(response)
    required_keys = ["disease", "treatment", "gene"]
    for key in required_keys:
        if key not in response:
            raise ValueError(f"Missing key: {key}")
        
    study_id = study_id = re.search(r".*/(.+)\.pdf", file_path)

    csv_file_path = "test_db.csv"
    try:
        df = pd.read_csv(csv_file_path)
        print("\nCSV file loaded successfully.")
    except FileNotFoundError:
        print("\nCSV file not found. Creating a new file.")
        df = pd.DataFrame(columns=["Study_identifier","disease",
                                "treatment","gene","ORDO_code","treatment_ID"])

    ordo_code = get_orpha_code(response["disease"])["OrphaCode"]
    chemi_id = list()
    #for treatment in response["treatment"]:
       #chemi_id.append(get_chemi_id(treatment))

    new_row = {
        "Study_identifier": study_id.group(1) if study_id else "None",
        "disease": response["disease"],
        "treatment": response["treatment"],
        "gene": response["gene"],  
        "ORDO_code": ordo_code if ordo_code else "None",
        "treatment_ID": chemi_id if chemi_id else "None"
        }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_file_path, index=False)
    print("\nRetrieval saved successfully.")

def validate_db(test_data, vali_data, output_file="output.txt"):
    with open(output_file, "w", encoding="utf-8") as file:  
        for identifier in test_data["Study_identifier"]:
            output = []  
            
            output.append(f"\nStudy Identifier: {identifier}")
            output.append("=" * 50)
            
            test_rows = test_data[test_data['Study_identifier'] == identifier].drop(columns=['Study_identifier'])
            vali_rows = vali_data[vali_data['Study_identifier'] == str(identifier)].drop(columns=['Study_identifier'])
            
            output.append("Test Data:")
            if not test_rows.empty:
                output.append(tabulate(test_rows, headers="keys", tablefmt="grid"))
            else:
                output.append("No test data available.")
            
            output.append("\nValidation Data:")
            if not vali_rows.empty:
                output.append(tabulate(vali_rows, headers="keys", tablefmt="grid"))
            else:
                output.append("No validation data available.")
            
            output.append("=" * 50)
            
            full_output = "\n".join(output)
            print(full_output)
            file.write(full_output + "\n")

file_path = "C:/Users/Adrian/Desktop/Bioinformatik/Projekt/Treatbolome/pdfs"
model = "qwen2.5:72b"

query = """What primary disease does the paper address? Classify it by type or 
            subtype, if applicable. What genes play a role in 
            this disease and what treatment options specifically linked to this
            are suggested? Ensure that the treatments are described with 
            precision, avoiding vague terms like 'dietary supplementation' or 
            'pharmaceutical therapy'. Provide detailed names of drugs, 
            therapies, or other specific interventions"""

for file in os.listdir(file_path):
    analyze_pdf(file_path +"/"+ file, model, query)
    

test_data = pd.read_csv("test_db.csv")
vali_data = pd.read_csv("validation_data.csv", sep=";")

validate_db(test_data, vali_data)

