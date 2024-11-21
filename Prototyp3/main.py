from pdf_retriever import PDFRetriever
import pandas as pd

file_path = "test.pdf"  
model = "llama3.2:latest"

analyzer = PDFRetriever(file_path, model)
analyzer.load_pdf()
analyzer.initialize_chain()

query = """What primary disease does the paper address, what genes play a role in 
            this disease and what treatment options specifically linked to this
            are suggested?"""

response = analyzer.analyze_document(query)
print(f"\n{model} response:")
print(response)
required_keys = ["disease", "treatment"]
for key in required_keys:
    if key not in response:
        raise ValueError(f"Missing key: {key}")

csv_file_path = "test_db.csv"
try:
    df = pd.read_csv(csv_file_path)
    print("\nCSV file loaded successfully.")
except FileNotFoundError:
    print("\nCSV file not found. Creating a new file.")
    df = pd.DataFrame(columns=["disease", "treatment", "gene"])

new_row = {
    "disease": response["disease"],
    "gene": response["gene"],  
    "treatment": response["treatment"]
    }

df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

df.to_csv(csv_file_path, index=False)
print("\nRetrieval saved successfully.")