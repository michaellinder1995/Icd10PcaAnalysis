# xml_parser.py
import xml.etree.ElementTree as ET
import re
from typing import Dict, List, Optional, Tuple

def clean_text(text: Optional[str]) -> str:
    """Bereinigt Text von überflüssigen Leerzeichen und Zeilenumbrüchen."""
    if text is None:
        return ""
    text = ' '.join(text.split())
    return text.strip()

def get_specific_rubric_text(element: Optional[ET.Element], kind: str) -> str:
    """Extrahiert den Text einer spezifischen Rubric-Art (z.B. 'preferred')."""
    if element is None:
        return ""
    rubric = element.find(f"./Rubric[@kind='{kind}']")
    if rubric is not None:
        label = rubric.find('./Label')
        if label is not None:
            return clean_text(ET.tostring(label, encoding='unicode', method='text'))
    return ""

def get_all_rubric_texts(element: Optional[ET.Element]) -> List[str]:
    """
    Extrahiert und bereinigt Texte aus relevanten Rubric-Kindern
    eines Class-Elements (für die Aggregation).
    """
    texts = []
    if element is None:
        return texts
    # Relevante Rubric-Arten für den Gesamttext
    relevant_kinds = ['preferred', 'definition', 'inclusion', 'text', 'coding-hint', 'exclusion'] # 'exclusion' auch? Ggf. entfernen.
    for rubric in element.findall('./Rubric'):
        kind = rubric.get('kind')
        if kind in relevant_kinds:
            label = rubric.find('./Label')
            if label is not None:
                rubric_text = clean_text(ET.tostring(label, encoding='unicode', method='text'))
                if rubric_text:
                    # Füge Art der Rubrik hinzu für Kontext (optional)
                    # texts.append(f"[{kind.upper()}] {rubric_text}")
                    texts.append(rubric_text)
    return texts

def parse_icd_xml_for_f_codes(xml_file_path: str) -> List[Dict[str, str]]:
    """
    Parses die ICD-10 GM XML (ClaML) Datei und extrahiert für 3-stellige
    F-Codes (F00-F99) den F-Code, den bevorzugten Titel (preferred label)
    und den aggregierten Beschreibungstext.

    Args:
        xml_file_path: Pfad zur ICD-10 GM XML Datei.

    Returns:
        Eine Liste von Dictionaries, jedes mit 'F_Code', 'PreferredLabel',
        und 'AggregatedText'. Gibt eine leere Liste zurück bei Fehlern.
    """
    parsed_data: List[Dict[str, str]] = []
    print(f"Lese und parse XML-Datei: {xml_file_path}")
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        print("XML erfolgreich geparst.")

        for i in range(100):
            f_code = f"F{i:02d}"
            # print(f"Verarbeite Code: {f_code}...") # Optional

            f_code_element = root.find(f".//Class[@code='{f_code}'][@kind='category']")

            if f_code_element is not None:
                # 1. Bevorzugten Titel (preferred label) des Fxx Elements extrahieren
                preferred_label = get_specific_rubric_text(f_code_element, 'preferred')
                if not preferred_label:
                    preferred_label = f"Kein Titel für {f_code}" # Fallback

                # 2. Aggregierten Text sammeln (Fxx + Fxx.y)
                aggregated_texts: List[str] = []
                aggregated_texts.extend(get_all_rubric_texts(f_code_element))

                subclass_codes = [sub.get('code') for sub in f_code_element.findall('./SubClass')]
                for sub_code in subclass_codes:
                    if sub_code:
                        sub_code_element = root.find(f".//Class[@code='{sub_code}'][@kind='category']")
                        if sub_code_element is not None:
                             aggregated_texts.extend(get_all_rubric_texts(sub_code_element))

                final_aggregated_text = " ".join(aggregated_texts)

                # Füge Eintrag zur Ergebnisliste hinzu, wenn Text vorhanden ist
                if final_aggregated_text:
                    parsed_data.append({
                        'F_Code': f_code,
                        'PreferredLabel': preferred_label,
                        'AggregatedText': final_aggregated_text
                    })
                # else:
                #      print(f"  Warnung: Keine aggregierten Texte für {f_code} gefunden.")

            # else:
            #     print(f"  Warnung: Hauptelement für {f_code} nicht gefunden.")

    except FileNotFoundError:
        print(f"FEHLER: Die Datei '{xml_file_path}' wurde nicht gefunden.")
        return []
    except ET.ParseError as e:
        print(f"FEHLER beim Parsen der XML-Datei: {e}")
        return []
    except Exception as e:
        print(f"FEHLER: Ein unerwarteter Fehler ist beim Parsen aufgetreten: {e}")
        return []

    print(f"Text-Extraktion abgeschlossen. {len(parsed_data)} F-Codes mit Text gefunden.")
    return parsed_data


# Optional: Testblock
if __name__ == '__main__':
    print("Führe Test des XML Parsers aus (neue Struktur)...")
    test_xml_path = 'icd10gm.xml'
    parsed_data = parse_icd_xml_for_f_codes(test_xml_path)

    if parsed_data:
        print(f"\nTest-Parsing erfolgreich. {len(parsed_data)} Einträge gefunden.")
        # Drucke erstes Ergebnis
        print("\nBeispielhafter erster Eintrag:")
        print(parsed_data[0])
        # Drucke Beispiel für F20
        f20_entry = next((item for item in parsed_data if item['F_Code'] == 'F20'), None)
        if f20_entry:
             print("\nEintrag für F20:")
             print(f"  F_Code: {f20_entry['F_Code']}")
             print(f"  PreferredLabel: {f20_entry['PreferredLabel']}")
             print(f"  AggregatedText (Anfang): {f20_entry['AggregatedText'][:300]}...")
    else:
        print("\nTest-Parsing fehlgeschlagen oder keine Daten gefunden.")