import xml_parser # Unser angepasster Parser

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import plotly.express as px # Importiere Plotly Express

# Optional: Für UMAP statt PCA
# from umap import UMAP

# --- Schritt 1: Daten laden durch Aufruf des Parsers ---
xml_file_path = 'icd10gm.xml'
# Gibt jetzt eine Liste von Dictionaries zurück
parsed_data = xml_parser.parse_icd_xml_for_f_codes(xml_file_path)

# --- Schritt 2: Daten vorbereiten UND ALS CSV SPEICHERN ---
if not parsed_data:
    print("FEHLER: Keine Daten aus der XML-Datei extrahiert. Analyse wird abgebrochen.")
    exit()

# DataFrame direkt aus der Liste von Dictionaries erstellen
df = pd.DataFrame(parsed_data)
# Stellt sicher, dass die Spalten die erwarteten Namen haben
# (sollten sie durch den Parser, aber sicher ist sicher)
df = df[['F_Code', 'PreferredLabel', 'AggregatedText']]
print(f"DataFrame erstellt mit {len(df)} Einträgen und Spalten: {df.columns.tolist()}")


# >>> Speichern des DataFrames als CSV-Datei (enthält jetzt PreferredLabel) <<<
csv_filename = 'icd_f_codes_parsed_full.csv'
try:
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    print(f"Die geparsten Daten wurden erfolgreich in '{csv_filename}' gespeichert.")
except Exception as e:
    print(f"FEHLER beim Speichern der CSV-Datei '{csv_filename}': {e}")


# Liste der Texte für das Embedding (nur der aggregierte Text wird benötigt)
texts_for_embedding = df['AggregatedText'].tolist()
# Liste der F-Codes (wird später für Labels/Hover Info benötigt)
f_codes = df['F_Code'].tolist()

# --- Schritt 3: Embeddings erstellen (wie zuvor) ---
print("Lade Embedding-Modell 'paraphrase-multilingual-mpnet-base-v2'...")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

print("Erstelle Embeddings für die Texte...")
embeddings = model.encode(texts_for_embedding, show_progress_bar=True)
print(f"Embeddings erstellt. Shape: {embeddings.shape}")

# --- Schritt 4: Dimensionsreduktion (wie zuvor) ---
print("Reduziere Dimensionen mit PCA auf 2 Komponenten...")
pca = PCA(n_components=2, random_state=42)
reduced_embeddings = pca.fit_transform(embeddings)

# Füge die reduzierten Koordinaten zum DataFrame hinzu
df['PCA1'] = reduced_embeddings[:, 0]
df['PCA2'] = reduced_embeddings[:, 1]

# --- Schritt 5: INTERAKTIVE Visualisierung mit Plotly ---
print("Erstelle interaktiven Plot mit Plotly...")

fig = px.scatter(df,
                 x='PCA1',          # Spalte für X-Achse
                 y='PCA2',          # Spalte für Y-Achse
                 text='F_Code',     # Zeige F-Code als Text neben dem Punkt (optional)
                 hover_name='F_Code', # Zeigt den F-Code fett im Hover-Tooltip an
                 hover_data={       # Definiert, was im Hover-Tooltip angezeigt wird
                     'F_Code': False, # Nicht nochmal anzeigen, da schon in hover_name
                     'PreferredLabel': True, # Zeige den Namen der Störung!
                     'PCA1': False,  # Zeige PCA1 formatiert an
                     'PCA2': False,
                    },   # Zeige PCA2 nicht an
                 title='Interaktive 2D Visualisierung der ICD-10 F-Codes (Kapitel V) - Hover für Details',
                 labels={'PCA1': 'PCA Komponente 1', 'PCA2': 'PCA Komponente 2'} # Achsenbeschriftungen
                )

# Konfiguriere die Textlabels neben den Punkten (kleiner und zentriert oben)
fig.update_traces(textposition='top center', textfont_size=9)

# Layout-Anpassungen (optional)
fig.update_layout(
    hovermode='closest', # Hover-Effekt für den nächsten Punkt
    xaxis=dict(gridcolor='lightgrey'),
    yaxis=dict(gridcolor='lightgrey'),
    plot_bgcolor='rgba(0,0,0,0)' # Transparenter Hintergrund

)

#speichere plot
html_filename = 'index.html'

try:
    fig.write_html(html_filename)
    print(f"Interaktiver Plot wurde als '{html_filename}' gespeichert.")
except Exception as e:
    print(f"FEHLER beim Speichern der HTML-Datei: {e}")

# Zeige den Plot (öffnet sich normalerweise im Browser oder im Notebook)
fig.show()

print("Analyse abgeschlossen. Interaktiver Plot sollte angezeigt werden.")

# Optional: DataFrame mit PCA-Ergebnissen speichern
# try:
#     df_results = df[['F_Code', 'PreferredLabel', 'PCA1', 'PCA2']] # Nur relevante Spalten
#     df_results.to_csv('icd10_f_codes_pca_results.csv', index=False, encoding='utf-8')
#     print("Ergebnisse inkl. PCA in 'icd10_f_codes_pca_results.csv' gespeichert.")
# except Exception as e:
#     print(f"Fehler beim Speichern der CSV-Datei mit PCA-Ergebnissen: {e}")