import json
from rapidfuzz import process

class Fuzzy_Orpha_Mapper:
    def __init__(self, input_file):
        self.context_data = self.load_json_file(input_file)
        self.simplified_context = self.simplify_data()
        self.diseases_data = self.simplified_context
        self.all_names = self.get_all_names()

    # Function to load the JSON file
    def load_json_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    # Function to save the JSON file
    def save_json_file(self, data, output_path):
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    # Simplifying the JSON data
    def simplify_data(self):
        simplified_context = []
        try:
            for disorder in self.context_data["JDBOR"][0]["DisorderList"][0]["Disorder"]:
                disorder_data = {
                    "name": disorder["Name"][0]["label"],
                    "OrphaCode": disorder["OrphaCode"]
                }

                # Extract synonyms if available
                synonyms = []
                if "SynonymList" in disorder and "Synonym" in disorder["SynonymList"][0]:
                    synonyms = [synonym["label"] for synonym in disorder["SynonymList"][0]["Synonym"]]

                if synonyms:
                    disorder_data["Synonyms"] = synonyms

                simplified_context.append(disorder_data)

            return simplified_context
        except KeyError as e:
            print(f"KeyError: {e} - Please check the JSON structure.")
        except IndexError as e:
            print(f"IndexError: {e} - Please check if the lists contain the expected elements.")
        return []

    # Extract disease names and synonyms
    def get_all_names(self):
        disease_names = [entry['name'] for entry in self.diseases_data]
        synonym_names = [synonym for entry in self.diseases_data if "Synonyms" in entry for synonym in entry["Synonyms"]]
        return disease_names + synonym_names

    # Function to find matching OrphaCodes using fuzzy matching
    def find_orpha_code(self, input_name, similarity_threshold=60):
        # Fuzzy matching with RapidFuzz (returns the best num_matches)
        matches = process.extract(input_name, self.all_names)

        for match in matches:
            matched_name = match[0]
            similarity_score = match[1]
            
            # If the similarity score is below the threshold, skip the match
            if similarity_score < similarity_threshold:
                continue
            
            # Find the entry for the matched name
            matched_entry = next(
                (item for item in self.diseases_data if item["name"] == matched_name), 
                None
            )
            
            if not matched_entry:
                matched_entry = next(
                    (item for item in self.diseases_data if "Synonyms" in item and matched_name in item["Synonyms"]), 
                    None
                )
            
            if matched_entry:
                # Output in the desired format
                if "Synonyms" in matched_entry and matched_name in matched_entry["Synonyms"]:
                    synonym = matched_name
                    main_name = matched_entry["name"]
                    # Return: OrphaCode, main name, synonym (if available)
                    return {"OrphaCode" : matched_entry['OrphaCode'], "Name" : main_name, "Synonym" : synonym}
                
                # If no synonym matched, return only the main name and OrphaCode
                return {"OrphaCode" : matched_entry['OrphaCode'], "Name" : matched_name}
        
        # If no match is found, return None
        return None