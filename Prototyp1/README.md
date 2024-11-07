# Erster RAG Prototyp

Moin Leute, das Python Skript nutzt eine Kombination aus einer PDF-Vektor-Datenbank und einem Retrieval-Augmented Generation (RAG)-Modell, um Antworten auf Benutzereingaben in Bezug auf den Inhalt des hochgeladenen PDFs zu generieren.

## Funktionen 

- **Modellauswahl**: Man kann das Modell auswählen, das man für die Analyse verwenden möchte.
- **PDF-Analyse**: Man kann ein PDF - Dokument hochladen und dessen Inhalt analysieren lassen..
- **Interaktive Abfrage**: Man stellt Fragen zum Inhalt des PDFs, und erhält Antworten von dem gewählten Modell.
- **Vector-Datenbank**: Das Programm verwendet eine Vektor-Datenbank zur effizienten Suche im PDF-Inhalt.
- **RAG-Chain**: Das Modell nutzt eine RAG-Chain, um auf die Informationen im PDF zuzugreifen und darauf basierend Antworten zu generieren.

## Installation 

Um das Programm auszuführen, benötigt ihr zunächst mindestens ein lokales LLM. Geht dafür auf [https://ollama.com/](https://ollama.com/) und ladet euch zunächst Ollama herunter, um die LLMs lokal auf euren Rechnern nutzen zu können. 

Anschließend könnt ihr Ollama von der Kommandozeile aus nutzen, um llama Modelle lokal zu speichern und zu verwenden. Mit folgendem Befehl zieht ihr euch llama3.1 (Größe 4.7 GB):

```bash
Ollama pull llama3.1
```

Anschließend könnt ihr euch mit python ein virtual environment erstellen und dann die Pakete installieren: 

```bash 
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Nutzung 

Ihr könnt das Programm über die Kommandozeile ausführen. Gebt dort den folgenden Befehl ein und passt die Parameter nach Bedarf an:

```bash
python rag_test.py -f <pfad\zur\pdf\datei> -m <Modellname>
```

### Parameter 

- `-f, --file` (Pflicht): Der Pfad zur PDF-Datei, die ihr analysieren wollt.

- `-m,--model` (Optional): Das Modell, das ihr für die Analyse verwenden möchten. Der Standardwert ist llama3.1:latest.

### Beispiel 
```bash
python rag_test.py -f dokument.pdf -m llama3.1:latest
```

