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
                # rstrip() zachowuje oryginalne wyrównanie i spacje wewnątrz linii danych
                data_lines.append(line.rstrip())

    # Pobieramy dokładnie ostatnich 20 rekordów
    last_20_records = data_lines[-20:]
    
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
    
    output_lines.extend(last_20_records)
    
    # Zapisujemy nowy plik w bezpiecznym kodowaniu UTF-8
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines) + '\n')


def process_entire_dataset(input_folder='dataset', output_folder='cleaned_dataset'):
    # Sprawdzenie czy folder wejściowy istnieje
    if not os.path.exists(input_folder):
        print(f"Błąd: Folder źródłowy '{input_folder}' nie istnieje!")
        return

    # Tworzenie folderu wyjściowego jeśli nie istnieje
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Pobranie wszystkich plików z folderu
    search_pattern = os.path.join(input_folder, '*')
    all_files = [f for f in glob.glob(search_pattern) if os.path.isfile(f)]
    
    print(f"Rozpoczęto przetwarzanie... Znaleziono {len(all_files)} plików w '{input_folder}'.")
    
    success = 0
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_folder, file_name)
        
        try:
            clean_grdc_file(file_path, output_path)
            success += 1
        except Exception as e:
            print(f" -> Błąd w pliku {file_name}: {str(e)}")
            
    print(f"\nSukces! Oczyszczono poprawnie {success} z {len(all_files)} plików.")
    print(f"Wyczyszczone dane zostały zapisane w folderze: '{output_folder}'")


if __name__ == '__main__':
    process_entire_dataset(input_folder='dataset', output_folder='cleaned_dataset')