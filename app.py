# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio
import os

# --- Konfiguration und Initialisierung ---
app = Flask(__name__)

# Pfade zu den vorberechneten Daten
CSV_FILE = 'icd_f_codes_parsed_full.csv'
EMBEDDINGS_FILE = 'icd_embeddings.npy'
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

# Globale Variablen zum Speichern der geladenen Daten und Modelle
try:
    print("Lade vorberechnete Daten...")
    df_icd = pd.read_csv(CSV_FILE)
    embeddings_icd = np.load(EMBEDDINGS_FILE)
    print(f"ICD-Daten ({len(df_icd)} Einträge) und Embeddings (Shape: {embeddings_icd.shape}) geladen.")

    print(f"Lade Embedding-Modell '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    print("Embedding-Modell geladen.")

    pca = PCA(n_components=2, random_state=42)
    print("PCA-Objekt initialisiert.")

    # Überprüfe Konsistenz
    if len(df_icd) != embeddings_icd.shape[0]:
        raise ValueError("Anzahl der Einträge in CSV und Embeddings stimmt nicht überein!")

except FileNotFoundError as e:
    print(f"FEHLER: Benötigte Datei nicht gefunden: {e}. Bitte zuerst 'precompute_embeddings.py' ausführen.")
    exit()
except Exception as e:
    print(f"FEHLER beim Initialisieren der Anwendung: {e}")
    exit()

# --- Routen / Endpunkte ---

@app.route('/')
def index():
    """Liefert die Haupt-HTML-Seite aus."""
    # Flask sucht automatisch im Ordner 'templates' nach 'index.html'
    return render_template('index.html')

@app.route('/get_initial_plot', methods=['GET'])
def get_initial_plot():
    """Berechnet und liefert den Plot nur mit ICD-10 Daten."""
    print("Berechne initialen Plot...")
    try:
        # PCA nur auf ICD-Daten anwenden
        pca_result_icd = pca.fit_transform(embeddings_icd)
        df_plot = df_icd.copy()
        df_plot['PCA1'] = pca_result_icd[:, 0]
        df_plot['PCA2'] = pca_result_icd[:, 1]
        df_plot['Type'] = 'ICD-10' # Typ für spätere Unterscheidung/Färbung

        fig = px.scatter(df_plot, x='PCA1', y='PCA2',
                         text='F_Code',
                         hover_name='F_Code',
                         hover_data={'F_Code': False, 'PreferredLabel': True, 'PCA1':':.2f', 'PCA2':':.2f'},
                         title='ICD-10 F-Codes (Kapitel V) - PCA Visualisierung')
        fig.update_traces(textposition='top center', textfont_size=9)
        fig.update_layout(hovermode='closest')

        # Konvertiere Plot in JSON, das von Plotly.js gelesen werden kann
        plot_json = pio.to_json(fig)
        return jsonify(success=True, plot_json=plot_json)

    except Exception as e:
        print(f"Fehler in /get_initial_plot: {e}")
        return jsonify(success=False, error=str(e)), 500


@app.route('/add_and_plot', methods=['POST'])
def add_and_plot():
    """Nimmt neue Daten entgegen, berechnet Embedding + PCA neu, liefert neuen Plot."""
    print("Empfange Daten für neuen Punkt...")
    try:
        data = request.get_json()
        person_name = data.get('person_name', 'Unbekannt')
        disease_name = data.get('disease_name', 'Neue Eingabe')
        disease_description = data.get('disease_description', '')

        if not disease_description:
            return jsonify(success=False, error="Krankheitsbeschreibung fehlt."), 400

        # 1. Embedding für neue Beschreibung berechnen
        print("Berechne Embedding für neue Beschreibung...")
        new_embedding = model.encode([disease_description]) # Muss als Liste übergeben werden

        # 2. Embeddings kombinieren
        combined_embeddings = np.vstack((embeddings_icd, new_embedding))
        print(f"Kombinierte Embeddings Shape: {combined_embeddings.shape}")

        # 3. PCA neu auf kombinierten Daten anwenden
        # WICHTIG: fit_transform hier, da sich die Daten geändert haben!
        print("Berechne PCA für kombinierte Daten neu...")
        pca_result_combined = pca.fit_transform(combined_embeddings)

        # 4. DataFrame für Plot vorbereiten
        # Labels und Typen erstellen
        labels = df_icd['F_Code'].tolist() + [f"{disease_name} ({person_name})"]
        hover_labels = df_icd['F_Code'].tolist() + [f"{disease_name}"] # Kürzer für hover_name
        preferred_labels = df_icd['PreferredLabel'].tolist() + [disease_description[:100] + "..."] # Beschreibung als Hover-Info
        types = ['ICD-10'] * len(df_icd) + ['Neue Eingabe']

        df_plot = pd.DataFrame({
            'Label': labels,
            'HoverLabel': hover_labels,
            'Description': preferred_labels, # Wiederverwendung der Spalte für Hover
            'Type': types,
            'PCA1': pca_result_combined[:, 0],
            'PCA2': pca_result_combined[:, 1]
        })

        # 5. Plotly Figur erstellen
        print("Erstelle Plotly Figur...")
        fig = px.scatter(df_plot, x='PCA1', y='PCA2',
                         text='Label',  # Zeigt F-Code oder neuen Namen an
                         color='Type',  # Färbt Punkte nach Typ (ICD-10 / Neue Eingabe)
                         hover_name='HoverLabel',
                         hover_data={'Label': False, # Nicht doppelt anzeigen
                                     'Description': True, # Zeigt PreferredLabel oder Beschreibung
                                     'PCA1':':.2f',
                                     'PCA2':':.2f'},
                         title=f'ICD-10 F-Codes mit "{disease_name}" - PCA Visualisierung')

        fig.update_traces(textposition='top center', textfont_size=9)
        fig.update_layout(hovermode='closest')

        # Plot als JSON zurücksenden
        plot_json = pio.to_json(fig)
        return jsonify(success=True, plot_json=plot_json)

    except Exception as e:
        print(f"Fehler in /add_and_plot: {e}")
        # Bei Fehlern ggf. den initialen Plot zurücksenden oder eine Fehlermeldung
        return jsonify(success=False, error=str(e)), 500

# --- Startet den Flask Development Server ---
if __name__ == '__main__':
    app.run()