import os
import glob

def clean_grdc_file(file_path, output_path):
    metadata = {
        "River": "Nieznana",
        "Station": "Nieznana",
        "Country": "Nieznany",
        "Latitude": "-999.000",
        "Longitude": "-999.000",
        "Catchment": "-999.00",
        "Altitude": "-999.00"
    }
    
    data_lines = []
    
    # Otwieramy plik z kodowaniem 'latin-1' (ISO-8859-1), aby bezpiecznie 
    # odczytać znaki DOS-ASCII (np. ² lub ˛) bez wywoływania błędu Crash.
    with open(file_path, 'r', encoding='latin-1') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Przetwarzanie nagłówków / komentarzy
        if stripped.startswith('#'):
            content = stripped[1:].strip()
            
            if "River:" in content:
                metadata["River"] = content.split("River:")[1].strip()
            elif "Station:" in content:
                metadata["Station"] = content.split("Station:")[1].strip()
            elif "Country:" in content:
                metadata["Country"] = content.split("Country:")[1].strip()
            elif "Latitude (DD):" in content:
                metadata["Latitude"] = content.split("Latitude (DD):")[1].strip()
            elif "Longitude (DD):" in content:
                metadata["Longitude"] = content.split("Longitude (DD):")[1].strip()
            elif "Catchment area" in content:
                val = content.split(":")[-1].strip()
                # Zabezpieczenie: jeśli wartość catchment area przeskoczyła do nowej linii
                if not val and i + 1 < len(lines):
                    next_line = lines[i+1].strip()
                    if next_line.startswith('#'):
                        next_line = next_line[1:].strip()
                    if next_line and ';' not in next_line:
                        val = next_line
                metadata["Catchment"] = val if val else "-999.00"
            elif "Altitude (m ASL):" in content:
                metadata["Altitude"] = content.split("Altitude (m ASL):")[1].strip()
                
        # Przetwarzanie linii z danymi liczbowymi
        else:
            # Sprawdzamy czy linia zawiera dane (jest separator i nie jest to nagłówek tabeli YYYY-MM-DD)
            if ';' in stripped and not stripped.startswith('YYYY'):
                parts = stripped.split(';')
                if parts:
                    date_part = parts[0].strip()
                    try:
                        # Wyciągamy rok z daty (YYYY-MM-DD)
                        year = int(date_part.split('-')[0])
                        # Zostawiamy tylko i wyłącznie lata 2023, 2024 oraz 2025
                        if year in [2023, 2024, 2025]:
                            # rstrip() zachowuje oryginalne wyrównanie i spacje wewnątrz linii danych
                            data_lines.append(line.rstrip())
                    except (ValueError, IndexError):
                        pass # Ignoruj uszkodzone wiersze

    # JEŚLI BRAK DANYCH Z LAT 2023-2025 - POMIJAMY GENEROWANIE PLIKU PLIKU
    if not data_lines:
        return False
    
    # Budujemy strukturę wyjściową dbając o idealne wyrównanie spacji ze wzoru
    output_lines = [
        f"River: {metadata['River']}",
        f"Station: {metadata['Station']}",
        f"Country: {metadata['Country']}",
        f"Latitude (DD):       {metadata['Latitude']}",
        f"Longitude (DD):      {metadata['Longitude']}",
        f"Catchment area (km²):      {metadata['Catchment']}",
        f"Altitude (m ASL):        {metadata['Altitude']}",
        "" # Pusta linia oddzielająca metadane od rekordów
    ]
    
    output_lines.extend(data_lines)
    
    # Zapisujemy nowy plik w bezpiecznym kodowaniu UTF-8
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines) + '\n')
        
    return True


def process_entire_dataset(input_folder='dataset', output_folder='cleaned_dataset'):
    # Sprawdzenie czy folder wejściowy istnieje
    if not os.path.exists(input_folder):
        print