# 🇪🇺 Analiza Danych Lotniczych - Europa/Polska

## Przegląd

Ten projekt został dostosowany do analizy danych europejskich i polskich linii lotniczych. Uwzględnia specyfikę rynku europejskiego, w tym:

- **Polskie lotniska i linie lotnicze** (WAW, KRK, GDN, LOT Polish Airlines)
- **Europejskich przewoźników** (Ryanair, Wizz Air, Lufthansa, KLM)
- **Specyficzne przyczyny opóźnień** (strajki, problemy lotniskowe)
- **Krótsze dystanse** charakterystyczne dla Europy
- **Sezonowość europejską** (warunki zimowe, ruch turystyczny)

## 🚀 Szybki Start

### 1. Wygeneruj Europejskie Dane

```python
from src.data_generator_eu import generate_european_data

# Generuj 50,000 europejskich lotów
df = generate_european_data(n_records=50000)
```

### 2. Uruchom Analizę Europejską

```python
from src.european_analysis import run_complete_european_analysis

# Kompletna analiza europejska
results = run_complete_european_analysis(df)
```

### 3. Demonstracja

```bash
python demo_european_analysis.py
```

## 📊 Dostępne Analizy

### Analiza Polskich Tras

- Statystyki połączeń z/do Polski
- Top 10 polskich tras lotniczych
- Porównanie polskich lotnisk
- Punktualność LOT vs konkurencja

### Analiza Europejskich Przewoźników

- Porównanie punktualności przewoźników
- Low-cost vs tradycyjne linie
- Analiza zasięgu tras
- Statystyki opóźnień według przewoźnika

### Europejskie Wzorce Pogodowe

- Sezonowość opóźnień pogodowych
- Najbardziej problematyczne regiony
- Porównanie Polska vs Europa
- Analiza warunków zimowych

### Analiza Strajków (Specyfika Europejska)

- Wpływ strajków na punktualność
- Kraje najbardziej dotknięte strajkami
- Przewoźnicy dotknięci strajkami
- Sezonowość strajków

## 🛠️ Konfiguracja dla Polski

### Dostosowanie Generatora Danych

W pliku `src/data_generator_eu.py`:

```python
# Zwiększ procent polskich lotów
if random.random() < 0.6:  # 60% lotów z/do Polski
    polish_airports = ['WAW', 'KRK', 'GDN', 'WRO', 'KTW', 'POZ']
    # ...

# Dodaj więcej polskich lotnisk
self.airports.update({
    'SZZ': 'Szczecin Goleniów',
    'BZG': 'Bydgoszcz',
    'IEG': 'Zielona Góra',
    'OSZ': 'Koszalin'
})
```

### Dostosowanie Analizy

W pliku `src/european_analysis.py`:

```python
# Dodaj specyficzne polskie analizy
def analyze_polish_seasonal_tourism(df):
    # Analiza ruchu turystycznego do Polski
    # Wakacje, majówka, sylwester, etc.
    pass

def analyze_polish_business_routes(df):
    # Analiza tras biznesowych (WAW-FRA, WAW-LHR)
    pass
```

## 📈 Przykładowe Wyniki

### Statystyki Polskie

```
🇵🇱 POLSKIE POŁĄCZENIA:
• Łączna liczba lotów: 18,432
• Procent wszystkich lotów: 36.9%
• Średnie opóźnienie: 12.3 min
• Punktualność: 73.5%

🛫 TOP POLSKIE TRASY:
1. WAW → LHR: 1,234 lotów, śr. opóźnienie 8.5 min
2. KRK → FRA: 987 lotów, śr. opóźnienie 11.2 min
3. GDN → ARN: 756 lotów, śr. opóźnienie 6.8 min
```

### Porównanie Przewoźników

```
✈️ PRZEWOŹNICY:
• LOT Polish Airlines: punktualność 78.2%
• Ryanair: punktualność 71.5%
• Wizz Air: punktualność 74.8%
• Lufthansa: punktualność 82.1%
```

## 🔧 Prawdziwe Dane

### Importowanie Rzeczywistych Danych

Aby użyć prawdziwych danych europejskich:

```python
# 1. Z pliku CSV
df = pd.read_csv('real_european_flights.csv')

# 2. Z API Flightradar24
import requests
# ... implementacja API

# 3. Z bazy danych Eurostat
# ... połączenie z bazą EU

# 4. Z oficjalnych danych ULC (Urząd Lotnictwa Cywilnego)
# ... polskie dane oficjalne
```

### Wymagane Kolumny

Twoje dane powinny zawierać:

```python
required_columns = [
    'flight_date',           # Data lotu
    'airline',              # Linia lotnicza
    'origin',               # Lotnisko origin (kod IATA)
    'destination',          # Lotnisko destination (kod IATA)
    'scheduled_departure',  # Planowany odlot
    'actual_departure',     # Rzeczywisty odlot
    'delay_minutes',        # Opóźnienie w minutach
    'delay_reason',         # Przyczyna opóźnienia
    'country_origin',       # Kraj origin
    'country_destination'   # Kraj destination
]
```

## 🌍 Europejskie Źródła Danych

### Oficjalne Źródła

- **Eurostat**: Oficjalne statystyki UE
- **EUROCONTROL**: Dane ruchu lotniczego
- **ULC Polska**: Urząd Lotnictwa Cywilnego
- **CAA UK**: Civil Aviation Authority
- **DFS Niemcy**: Deutsche Flugsicherung

### API i Usługi

- **Flightradar24 API**: Dane czasu rzeczywistego
- **OpenSky Network**: Otwarte dane lotnicze
- **FlightAware**: Dane comercyjne
- **Aviation Edge**: API lotnicze

### Przykład Integracji z API

```python
def fetch_european_data_from_api():
    # Przykład pobierania z OpenSky
    import requests

    url = "https://opensky-network.org/api/flights/departure"
    params = {
        'airport': 'EPWA',  # Warszawa Chopin
        'begin': '2024-01-01 00:00:00',
        'end': '2024-01-31 23:59:59'
    }

    response = requests.get(url, params=params)
    return response.json()
```

## 📊 Wizualizacje

### Mapy Europejskie

```python
import plotly.graph_objects as go

def create_european_route_map(df):
    # Mapa tras europejskich z fokusem na Polskę
    fig = go.Figure()

    # Dodaj polskie lotniska
    fig.add_trace(go.Scattergeo(
        lon=[21.0122, 19.7848],  # WAW, KRK
        lat=[52.2297, 50.0647],
        text=['Warszawa', 'Kraków'],
        mode='markers+text',
        marker_color='red',
        marker_size=10
    ))

    fig.update_geos(
        scope='europe',
        showland=True,
        landcolor='rgb(243, 243, 243)',
        coastlinecolor='rgb(204, 204, 204)',
    )

    return fig
```

## 🎯 Analiza Biznesowa

### KPI dla Polski

- **Punktualność polskich lotów**
- **Market share LOT vs konkurencja**
- **Popularność tras biznesowych vs turystycznych**
- **Sezonowość ruchu do/z Polski**
- **Wpływ pogody na polskie lotniska**

### Benchmarking Europejski

- **Porównanie z krajami V4**
- **Pozycja Polski w rankingu punktualności**
- **Analiza konkurencyjności hubów**
- **Wpływ low-cost na rynek polski**

## 🚨 Troubleshooting

### Częste Problemy

1. **Błąd importu matplotlib**

   ```bash
   pip install matplotlib seaborn plotly
   ```

2. **Brak danych polskich**

   - Zwiększ parametr `polish_flight_ratio` w generatorze
   - Sprawdź mapowanie krajów w `get_country()`

3. **Błędne kody lotnisk**

   - Użyj standardowych kodów IATA (WAW, KRK, GDN)
   - Sprawdź słownik `airports` w generatorze

4. **Problemy z kodowaniem**
   ```python
   pd.read_csv('file.csv', encoding='utf-8')
   ```

## 📝 Dalszy Rozwój

### Planowane Funkcje

- [ ] Integracja z prawdziwymi API
- [ ] Analiza kosztów opóźnień
- [ ] Predykcja opóźnień ML
- [ ] Dashboard interaktywny
- [ ] Raporty automatyczne
- [ ] Alerty punktualności

### Wkład w Projekt

1. Fork repository
2. Dodaj nowe funkcje europejskie
3. Testuj z prawdziwymi danymi
4. Utwórz pull request

## 📞 Wsparcie

Jeśli masz problemy z konfiguracją europejską:

1. Sprawdź przykłady w `demo_european_analysis.py`
2. Przeczytaj komentarze w kodzie
3. Przetestuj z małymi danymi (1000 rekordów)
4. Zweryfikuj kody lotnisk i krajów

---

**Powodzenia w analizie europejskich danych lotniczych! 🛫🇪🇺**
