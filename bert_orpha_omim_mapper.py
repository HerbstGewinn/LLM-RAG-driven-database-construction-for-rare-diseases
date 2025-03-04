import json
import re
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class BertOrphaOmimMapper:
    def __init__(self, input_file, model_name="all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device, model_kwargs={"torch_dtype": torch.float16})
        
        self.context_data = self.load_json_file(input_file)
        self.diseases_data = self.simplify_data()
        self.all_names = self.get_all_names()
        self.embeddings = self.embed_names(self.all_names)

    @staticmethod
    def load_json_file(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Fehler beim Laden der JSON-Datei: {e}")
            return {}

    @staticmethod
    def preprocess_name(name):
        stopwords = {"disease", "syndrome", "disorder", "type", "form", "variant"}
        name = re.sub(r"[^\w\s]", "", name)  
        words = name.lower().split()
        return " ".join([word for word in words if word not in stopwords])

    def simplify_data(self):
        simplified_context = []
        try:
            for disorder in self.context_data.get("JDBOR", [{}])[0].get("DisorderList", [{}])[0].get("Disorder", []):
                disorder_data = {
                    "name": disorder["Name"][0]["label"],
                    "OrphaCode": str(disorder["OrphaCode"]).strip(),
                    "Synonyms": [syn["label"] for syn in disorder.get("SynonymList", [{}])[0].get("Synonym", [])]
                }
                simplified_context.append(disorder_data)
            return simplified_context
        except (KeyError, IndexError) as e:
            print(f"Fehler beim Vereinfachen der JSON-Daten: {e}")
            return []

    def get_all_names(self):
        return [entry["name"] for entry in self.diseases_data] + [
            syn for entry in self.diseases_data for syn in entry.get("Synonyms", [])
        ]

    def embed_names(self, names):
        preprocessed_names = [self.preprocess_name(name) for name in names]
        return self.model.encode(preprocessed_names, convert_to_tensor=True)

    def find_orpha_code(self, input_name, threshold=0.6):
        input_embedding = self.model.encode([self.preprocess_name(input_name)], convert_to_tensor=True)
        similarities = cosine_similarity(input_embedding.cpu(), self.embeddings.cpu())[0]
        
        best_match_idx = similarities.argmax()
        if similarities[best_match_idx] < threshold:
            return None
        
        matched_name = self.all_names[best_match_idx]
        for entry in self.diseases_data:
            if matched_name == entry["name"] or matched_name in entry.get("Synonyms", []):
                return {"OrphaCode": entry["OrphaCode"], "Name": entry["name"], "Synonym": matched_name}
        return None

    @staticmethod
    def extract_clean_name_and_code(disease_name):
        match = re.match(r"^(.*?), (\d{6})", disease_name)
        return match.groups() if match else (disease_name, None)

    def find_omim_code(self, matched_name, file_path="morbidmap.txt", threshold=0.6):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                lines = [line.split("\t")[0] for line in file if not line.startswith("#") and "\t" in line]

            processed_names = [self.preprocess_name(name) for name in lines]
            disease_embeddings = self.model.encode(processed_names, convert_to_tensor=True)
            input_embedding = self.model.encode([self.preprocess_name(matched_name)], convert_to_tensor=True)

            similarities = cosine_similarity(input_embedding.cpu(), disease_embeddings.cpu())[0]
            best_match_idx = similarities.argmax()

            if similarities[best_match_idx] >= threshold:
                matched_name, omim_code = self.extract_clean_name_and_code(lines[best_match_idx])
                return {"OMIM_Code": omim_code, "Matched_Name": matched_name}
        except (FileNotFoundError, IndexError) as e:
            print(f"Fehler beim Laden der OMIM-Daten: {e}")
        return None

    def find_orpha_omim(self, disease_name, threshold=0.6):
        orpha_result = self.find_orpha_code(disease_name, threshold)
        if not orpha_result:
            return None

        matched_name = orpha_result["Name"]
        synonyms = [matched_name] + next(
            (entry["Synonyms"] for entry in self.diseases_data if entry["OrphaCode"] == orpha_result["OrphaCode"]),
            []
        )

        best_omim_result = None
        highest_score = 0
        for synonym in synonyms:
            omim_result = self.find_omim_code(synonym)
            if omim_result:
                match_score = cosine_similarity(
                    self.model.encode([self.preprocess_name(synonym)], convert_to_tensor=True).cpu(),
                    self.model.encode([self.preprocess_name(omim_result["Matched_Name"])], convert_to_tensor=True).cpu()
                )[0][0]

                if match_score > highest_score:
                    highest_score = match_score
                    best_omim_result = omim_result

        if best_omim_result:
            return {
                "Orpha_Code": orpha_result["OrphaCode"],
                "Orpha_Name": orpha_result["Name"],
                "OMIM_Code": best_omim_result["OMIM_Code"],
                "Omim_Name": best_omim_result["Matched_Name"]
            }

        return {
            "Orpha_Code": orpha_result["OrphaCode"],
            "Orpha_Name": orpha_result["Name"],
            "OMIM_Code": "Not found",
            "Omim_Name": "Not found"
        }
