import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Bert_Orpha_Mapper:
    def __init__(self, input_file, model_name='all-MiniLM-L6-v2'):
        self.context_data = self.load_json_file(input_file)
        self.simplified_context = self.simplify_data()
        self.diseases_data = self.simplified_context
        self.all_names = self.get_all_names()
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.embed_names(self.all_names)

    # Function to load the JSON file
    def load_json_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    # Function to save the JSON file
    def save_json_file(self, data, output_path):
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    # Preprocessing of names
    def preprocess_name(self, name):
        """Removes stopwords, special characters, and handles case insensitivity."""
        stopwords = {"disease", "syndrome", "disorder", "type", "form", "variant"}
        name = re.sub(r"[^\w\s]", "", name)  # Remove special characters
        words = name.split()
        filtered_words = [word for word in words if word.lower() not in stopwords]  # Remove stopwords
        return " ".join(filtered_words).lower()

    # Simplify the JSON data
    def simplify_data(self):
        simplified_context = []
        try:
            for disorder in self.context_data["JDBOR"][0]["DisorderList"][0]["Disorder"]:
                disorder_data = {
                    "name": disorder["Name"][0]["label"],
                    "OrphaCode": str(disorder["OrphaCode"]).strip(),
                    "OMIM_Codes": []
                }

                # Extract synonyms if available
                synonyms = []
                if "SynonymList" in disorder and "Synonym" in disorder["SynonymList"][0]:
                    synonyms = [synonym["label"] for synonym in disorder["SynonymList"][0]["Synonym"]]

                if synonyms:
                    disorder_data["Synonyms"] = synonyms

                # Extract OMIM codes (or other external references)
                if "ExternalReferenceList" in disorder and "ExternalReference" in disorder["ExternalReferenceList"][0]:
                    omim_codes = [
                        ref["Reference"]
                        for ref in disorder["ExternalReferenceList"][0]["ExternalReference"]
                        if ref["Source"] == "OMIM"
                    ]
                    disorder_data["OMIM_Codes"] = omim_codes

                simplified_context.append(disorder_data)

            return simplified_context
        except KeyError as e:
            print(f"KeyError: {e} - Please check the JSON structure.")
        except IndexError as e:
            print(f"IndexError: {e} - Please check if the lists contain the expected elements.")
        return []
    # Extract all disease names and synonyms
    def get_all_names(self):
        disease_names = [entry['name'] for entry in self.diseases_data]
        synonym_names = [synonym for entry in self.diseases_data if "Synonyms" in entry for synonym in entry["Synonyms"]]
        return disease_names + synonym_names

    # Generate embeddings for all names
    def embed_names(self, names):
        preprocessed_names = [self.preprocess_name(name) for name in names]
        return self.model.encode(preprocessed_names)

    # Find OrphaCodes using embedding-based matching
    def find_orpha_code(self, input_name, similarity_threshold=0.6):
        input_name_preprocessed = self.preprocess_name(input_name)
        input_embedding = self.model.encode([input_name_preprocessed])

        # Compute similarities
        similarities = cosine_similarity(input_embedding, self.embeddings)
        similarity_scores = list(zip(self.all_names, similarities[0]))

        # Sort results and find the best match
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        for matched_name, score in sorted_scores:
            if score < similarity_threshold:
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
                result = {
                    "OrphaCode": matched_entry["OrphaCode"],
                    "Name": matched_entry["name"],
                    "OMIM_Codes": matched_entry.get("OMIM_Codes", [])
                }

                if "Synonyms" in matched_entry and matched_name in matched_entry["Synonyms"]:
                    result["Synonym"] = matched_name

                return result

        return None
