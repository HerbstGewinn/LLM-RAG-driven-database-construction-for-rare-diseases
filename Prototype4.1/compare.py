import csv
#from tabulate import tabulate

# Dateien einlesen
def read_csv(file_path, delimiter=','):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        for row in reader:
            data.append({
                'Study_identifier': row['Study_identifier'],
                'disease': row.get('disease', ''),
                'ORDO_code': row.get('ORDO_code', ''),
                'treatment': row.get('treatment', ''),
                'gene': row.get('gene', ''),
                'treatment_ID': row.get('treatment_ID', '')
            })
    return data

def compare_csv(file1_data, file2_data, validation_data):
    comparison = []
    validation_dict = {row['Study_identifier']: row for row in validation_data}

    file2_dict = {row['Study_identifier']: row for row in file2_data}

    for row1 in file1_data:
        study_id = row1['Study_identifier']
        row2 = file2_dict.get(study_id)
        validation_row = validation_dict.get(study_id)

        if row2:
            validation_info = (f"Disease: {validation_row['disease']} | Treatment: {validation_row['treatment']} | Gene: {validation_row['gene']} | ORDO Code: {validation_row['ORDO_code']} | Treatment ID: {validation_row['treatment_ID']}"
                               if validation_row else "No validation data available")
            comparison.append({
                'Study_identifier': study_id,
                'File 1': f"Study Identifier: {row1['Study_identifier']} | Disease: {row1['disease']} | Treatment: {row1['treatment']} | Gene: {row1['gene']} | ORDO Code: {row1['ORDO_code']} | Treatment ID: {row1['treatment_ID']}",
                'File 2': f"Study Identifier: {row2['Study_identifier']} | Disease: {row2['disease']} | Treatment: {row2['treatment']} | Gene: {row2['gene']} | ORDO Code: {row2['ORDO_code']} | Treatment ID: {row2['treatment_ID']}",
                'Validation': validation_info
            })

    # List validation entries not in file1 or file2
    unmatched_validation = [row for row in validation_data if row['Study_identifier'] not in file2_dict and row['Study_identifier'] not in {row['Study_identifier'] for row in file1_data}]

    return comparison, unmatched_validation

def main():
    file1_path = 'reduced_test_db_no_reranker.csv'  # Pfad zur Datei 1
    file2_path = 'reduced_test_db_reranker.csv'  # Pfad zur Datei 2
    validation_path = 'validation_data.csv'  # Pfad zur Validierungsdatei

    # Dateien einlesen
    file1_data = read_csv(file1_path)
    file2_data = read_csv(file2_path)
    validation_data = read_csv(validation_path, delimiter=';')

    # Dateien vergleichen
    comparison_result, unmatched_validation = compare_csv(file1_data, file2_data, validation_data)

    # Vergleichsergebnisse ausgeben
    for row in comparison_result:
        print(f"Study Identifier: {row['Study_identifier']}")
        print(f"  No Reranker -> {row['File 1']}")
        print(f"  Reranker -> {row['File 2']}")
        print(f"  Validation -> {row['Validation']}")
        print("-")

    # Nicht gematchte Validation-Einträge ausgeben
    print("\nUnmatched Validation Entries:")
    for row in unmatched_validation:
        print(f"Study Identifier: {row['Study_identifier']} | Disease: {row['disease']} | Treatment: {row['treatment']} | Gene: {row['gene']} | ORDO Code: {row['ORDO_code']} | Treatment ID: {row['treatment_ID']}")

if __name__ == "__main__":
    main()
