# precompute_embeddings.py
import xml_parser # Unser XML-Parser-Modul
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# --- Konfiguration ---
XML_FILE = 'icd10gm.xml'
OUTPUT_CSV_FILE = 'icd_f_codes_parsed_full.csv'
OUTPUT_EMBEDDINGS_FILE = 'icd_embeddings.npy' # Endung .npy für NumPy Array
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

# --- 1. Daten parsen ---
print(f"Parse XML '{XML_FILE}'...")
# Gibt Liste von Dicts zurück: [{'F_Code': ..., 'PreferredLabel': ..., 'AggregatedText': ...}, ...]
parsed_data = xml_parser.parse_icd_xml_for_f_codes(XML_FILE)

if not parsed_data:
    print("FEHLER: Keine Daten aus XML extrahiert. Abbruch.")
    exit()

# --- 2. DataFrame erstellen und als CSV speichern ---
df = pd.DataFrame(parsed_data)
# Sicherstellen, dass die Spalten in definierter Reihenfolge sind
df = df[['F_Code', 'PreferredLabel', 'AggregatedText']]

try:
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
    print(f"Geparste Daten erfolgreich in '{OUTPUT_CSV_FILE}' gespeichert.")
except Exception as e:
    print(f"FEHLER beim Speichern der CSV-Datei '{OUTPUT_CSV_FILE}': {e}")
    exit()

# --- 3. Embeddings berechnen ---
print(f"Lade Embedding-Modell '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)

texts_for_embedding = df['AggregatedText'].tolist()

print(f"Erstelle Embeddings für {len(texts_for_embedding)} Texte...")
embeddings = model.encode(texts_for_embedding, show_progress_bar=True)
print(f"Embeddings erstellt. Shape: {embeddings.shape}")

# --- 4. Embeddings speichern ---
try:
    np.save(OUTPUT_EMBEDDINGS_FILE, embeddings)
    print(f"Embeddings erfolgreich in '{OUTPUT_EMBEDDINGS_FILE}' gespeichert.")
except Exception as e:
    print(f"FEHLER beim Speichern der Embeddings-Datei '{OUTPUT_EMBEDDINGS_FILE}': {e}")
    exit()

print("\nVorberechnung abgeschlossen.")