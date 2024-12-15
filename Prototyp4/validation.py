from tabulate import tabulate
import pandas as pd


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