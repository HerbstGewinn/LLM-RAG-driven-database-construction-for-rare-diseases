import pandas as pd

# Original CSV-Datei laden
df = pd.read_csv("test_db_reranker.csv")

# Nur die gewünschten Spalten beibehalten
filtered_df = df[["Study_identifier", "disease", "treatment", "gene", "ORDO_code","treatment_ID"]]

# Numerische und alphabetische Sortierung vorbereiten
def sort_key(identifier):
    identifier = str(identifier)  # Convert to string to handle both numbers and strings
    if identifier[0].isdigit():
        return (0, int(''.join(filter(str.isdigit, identifier))))
    else:
        return (1, identifier)

# Sortieren der DataFrame nach der benutzerdefinierten Reihenfolge
filtered_df = filtered_df.sort_values(by="Study_identifier", key=lambda col: col.map(sort_key))

# Neue CSV-Datei speichern
filtered_df.to_csv("reduced_test_db_reranker.csv", index=False)

print("Die Datei wurde erfolgreich gefiltert, sortiert und gespeichert.")
