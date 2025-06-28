#!/usr/bin/env python3
"""
Demonstracja analizy danych europejskich/polskich dla lotnictwa.
Pokazuje jak używać nowych narzędzi do analizy ruchu europejskiego.
"""

import pandas as pd
import sys
import os

# Dodaj src do ścieżki
sys.path.append('src')

from data_generator_eu import generate_european_data, EuropeanFlightDataGenerator
from european_analysis import run_complete_european_analysis
import utils

def main():
    """Główna funkcja demonstracyjna"""
    print("🇪🇺 DEMONSTRACJA ANALIZY EUROPEJSKICH DANYCH LOTNICZYCH")
    print("="*70)
    
    print("Krok 1: Generowanie europejskich danych...")
    
    # Generuj mniejszy dataset dla demonstracji (10,000 rekordów)
    df_european = generate_european_data(
        filename="data/raw/demo_european_flights.csv", 
        n_records=10000
    )
    
    print(f"\n✅ Wygenerowano {len(df_european)} europejskich rekordów")
    
    # Pokaż podstawowe informacje o danych
    print(f"\n📊 PODSTAWOWE STATYSTYKI:")
    print(f"   • Okres danych: {df_european['flight_date'].min()} do {df_european['flight_date'].max()}")
    print(f"   • Liczba unikalnych tras: {(df_european['origin'] + '-' + df_european['destination']).nunique()}")
    print(f"   • Liczba przewoźników: {df_european['airline'].nunique()}")
    print(f"   • Liczba krajów: {df_european['country_origin'].nunique()}")
    
    # Pokaż top kraje
    print(f"\n🌍 Top 5 krajów według liczby lotów (origin):")
    country_counts = df_european['country_origin'].value_counts().head(5)
    for i, (country, count) in enumerate(country_counts.items(), 1):
        pct = count / len(df_european) * 100
        print(f"   {i}. {country}: {count:,} lotów ({pct:.1f}%)")
    
    # Pokaż top przewoźników
    print(f"\n✈️ Top 5 przewoźników:")
    airline_counts = df_european['airline'].value_counts().head(5)
    for i, (airline, count) in enumerate(airline_counts.items(), 1):
        pct = count / len(df_european) * 100
        print(f"   {i}. {airline}: {count:,} lotów ({pct:.1f}%)")
    
    # Dodaj dodatkowe kolumny potrzebne do analizy
    print(f"\nKrok 2: Przygotowanie danych do analizy...")
    df_european = prepare_data_for_analysis(df_european)
    
    # Uruchom kompletną analizę europejską
    print(f"\nKrok 3: Uruchamianie kompletnej analizy europejskiej...")
    print("="*70)
    
    try:
        results = run_complete_european_analysis(df_european)
        print(f"\n✅ Analiza zakończona pomyślnie!")
        
        # Zapisz wyniki
        save_analysis_results(results, df_european)
        
    except Exception as e:
        print(f"❌ Błąd podczas analizy: {e}")
        print("Sprawdź czy masz zainstalowane wszystkie wymagane biblioteki:")
        print("pip install matplotlib seaborn plotly pandas numpy scipy")
    
    print(f"\n🎯 PODSUMOWANIE DEMONSTRACJI:")
    print(f"   • Wygenerowano i przeanalizowano {len(df_european):,} europejskich lotów")
    print(f"   • Przeanalizowano {df_european['country_origin'].nunique()} krajów europejskich")
    print(f"   • Szczególny fokus na polskie połączenia")
    print(f"   • Uwzględniono specyfikę europejską (strajki, krótsze dystanse)")
    print(f"   • Dane zapisane w: data/raw/demo_european_flights.csv")
    
    return df_european

def prepare_data_for_analysis(df):
    """Przygotowuje dane do analizy - dodaje brakujące kolumny"""
    
    # Konwertuj datę
    df['flight_date'] = pd.to_datetime(df['flight_date'])
    
    # Dodaj kolumny czasowe jeśli nie istnieją
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['flight_date'].dt.dayofweek
    
    if 'month' not in df.columns:
        df['month'] = df['flight_date'].dt.month
    
    if 'hour' not in df.columns:
        df['hour'] = pd.to_datetime(df['scheduled_departure'], format='%H:%M').dt.hour
    
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    # Kategorie opóźnień
    if 'delay_category' not in df.columns:
        df['delay_category'] = df['delay_minutes'].apply(categorize_delay)
    
    # Trasy
    if 'route' not in df.columns:
        df['route'] = df['origin'] + ' → ' + df['destination']
    
    return df

def categorize_delay(minutes):
    """Kategoryzuje opóźnienia"""
    if minutes == 0:
        return 'Punktualny'
    elif minutes <= 15:
        return 'Małe opóźnienie (1-15 min)'
    elif minutes <= 60:
        return 'Średnie opóźnienie (16-60 min)'
    else:
        return 'Duże opóźnienie (>60 min)'

def save_analysis_results(results, df):
    """Zapisuje wyniki analizy do plików"""
    
    # Stwórz folder na wyniki jeśli nie istnieje
    os.makedirs('results/european_analysis', exist_ok=True)
    
    try:
        # Zapisz podstawowe statystyki
        if 'summary' in results and results['summary']:
            summary_stats = {
                'total_flights': len(df),
                'avg_delay': df['delay_minutes'].mean(),
                'punctuality_pct': (df['delay_minutes'] == 0).mean() * 100,
                'avg_distance_km': df['distance_km'].mean(),
                'polish_flights': len(df[(df['country_origin'] == 'Polska') | 
                                        (df['country_destination'] == 'Polska')])
            }
            
            with open('results/european_analysis/summary_stats.txt', 'w', encoding='utf-8') as f:
                f.write("EUROPEJSKA ANALIZA LOTNICZA - PODSUMOWANIE\n")
                f.write("="*50 + "\n\n")
                for key, value in summary_stats.items():
                    f.write(f"{key}: {value}\n")
        
        print(f"   • Wyniki zapisane w: results/european_analysis/")
        
    except Exception as e:
        print(f"   ⚠️ Nie udało się zapisać wszystkich wyników: {e}")

def generate_sample_real_data():
    """
    Generuje przykładowe dane na podstawie rzeczywistych tras europejskich.
    Możesz to zastąpić prawdziwymi danymi z API lub plików CSV.
    """
    print("🔄 Alternatywnie: Generowanie danych na podstawie rzeczywistych tras...")
    
    # Przykład jak można by importować rzeczywiste dane
    real_routes_example = [
        # Format: (origin, destination, airline, typical_delay_min)
        ('WAW', 'LHR', 'LOT Polish Airlines', 8),
        ('WAW', 'FRA', 'Lufthansa', 12),
        ('KRK', 'LHR', 'Ryanair', 15),
        ('GDN', 'ARN', 'SAS', 6),
        ('WAW', 'CDG', 'Air France', 18),
        # ... dodaj więcej tras
    ]
    
    print(f"   • Przykład: {len(real_routes_example)} rzeczywistych tras zdefiniowanych")
    print("   • Aby użyć prawdziwych danych, zastąp funkcję generate_sample_real_data()")
    print("   • Możliwe źródła: Flightradar24 API, OpenSky Network, oficjalne dane lotnisk")
    
    return real_routes_example

if __name__ == "__main__":
    print("🚀 Uruchamianie demonstracji europejskiej analizy lotniczej...")
    
    try:
        df_result = main()
        
        print(f"\n📝 DALSZE KROKI:")
        print("1. Sprawdź wygenerowane wykresy i analizy")
        print("2. Dostosuj parametry w src/data_generator_eu.py do swoich potrzeb")
        print("3. Zmodyfikuj analizę w src/european_analysis.py")
        print("4. Dodaj prawdziwe dane europejskie zastępując generator")
        print("5. Uruchom pełną analizę z większą ilością danych (50,000+)")
        
        print(f"\n🔧 KONFIGURACJA DANYCH POLSKICH:")
        print("   • Zmień procent polskich lotów w EuropeanFlightDataGenerator")
        print("   • Dodaj więcej polskich lotnisk (RZE, LUZ, SZZ)")
        print("   • Dostosuj przyczyny opóźnień do polskiej specyfiki")
        print("   • Uwzględnij sezonowość ruchu turystycznego")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Przerwano przez użytkownika")
    except Exception as e:
        print(f"\n❌ Błąd: {e}")
        print("Sprawdź czy wszystkie zależności są zainstalowane:")
        print("pip install -r requirements.txt") 