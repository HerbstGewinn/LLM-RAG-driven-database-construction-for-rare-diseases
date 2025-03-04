from fuzzy_orpha_mapper import Fuzzy_Orpha_Mapper

disease_processor = Fuzzy_Orpha_Mapper('en_product1.json')

# User input and search function
input_disease = input("Please enter a disease name: ")

# Find OrphaCode and return it
orpha_code = disease_processor.find_orpha_code(input_disease)
if orpha_code:
    print(orpha_code)
else:
    print(f"No OrphaCode found for '{input_disease}'.")
