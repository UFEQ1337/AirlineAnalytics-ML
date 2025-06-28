"""
Moduł do analizy europejskich wzorców opóźnień lotniczych.
Dostosowany do specyfiki ruchu lotniczego w Europie z fokusem na Polskę.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja polskich fontów
plt.rcParams['font.family'] = ['DejaVu Sans']

def analyze_polish_routes(df):
    """Analiza tras z i do Polski"""
    print("🇵🇱 ANALIZA POLSKICH TRAS LOTNICZYCH")
    print("="*60)
    
    # Filtruj loty z/do Polski
    polish_flights = df[(df['country_origin'] == 'Polska') | 
                       (df['country_destination'] == 'Polska')].copy()
    
    print(f"📊 Statystyki polskich połączeń:")
    print(f"   • Łączna liczba lotów: {len(polish_flights):,}")
    print(f"   • Procent wszystkich lotów: {len(polish_flights)/len(df)*100:.1f}%")
    print(f"   • Średnie opóźnienie: {polish_flights['delay_minutes'].mean():.1f} min")
    print(f"   • Punktualność: {(polish_flights['delay_minutes'] == 0).mean()*100:.1f}%")
    
    # Top trasy z Polski
    polish_routes = polish_flights.copy()
    polish_routes['route'] = polish_routes['origin'] + ' → ' + polish_routes['destination']
    top_routes = polish_routes['route'].value_counts().head(10)
    
    print(f"\n🛫 Top 10 polskich tras:")
    for i, (route, count) in enumerate(top_routes.items(), 1):
        route_data = polish_flights[polish_flights['origin'] + ' → ' + polish_flights['destination'] == route]
        avg_delay = route_data['delay_minutes'].mean()
        print(f"   {i:2d}. {route}: {count:,} lotów, śr. opóźnienie {avg_delay:.1f} min")
    
    # Analiza polskich lotnisk
    polish_airports = ['WAW', 'KRK', 'GDN', 'WRO', 'KTW', 'POZ', 'RZE', 'LUZ']
    airport_stats = []
    
    for airport in polish_airports:
        airport_flights = df[(df['origin'] == airport) | (df['destination'] == airport)]
        if len(airport_flights) > 0:
            stats = {
                'Lotnisko': airport,
                'Liczba_lotów': len(airport_flights),
                'Średnie_opóźnienie': airport_flights['delay_minutes'].mean(),
                'Punktualność_%': (airport_flights['delay_minutes'] == 0).mean() * 100
            }
            airport_stats.append(stats)
    
    airport_df = pd.DataFrame(airport_stats).sort_values('Liczba_lotów', ascending=False)
    print(f"\n🏢 Statystyki polskich lotnisk:")
    for _, row in airport_df.iterrows():
        print(f"   • {row['Lotnisko']}: {row['Liczba_lotów']:,} lotów, "
              f"śr. opóźnienie {row['Średnie_opóźnienie']:.1f} min, "
              f"punktualność {row['Punktualność_%']:.1f}%")
    
    # Wizualizacja
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Top trasy
    top_routes.head(8).plot(kind='barh', ax=ax1, color='lightblue')
    ax1.set_title('Top 8 polskich tras', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Liczba lotów')
    
    # Porównanie polskich lotnisk
    airport_df.set_index('Lotnisko')['Średnie_opóźnienie'].plot(kind='bar', ax=ax2, color='orange')
    ax2.set_title('Średnie opóźnienia - polskie lotniska', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Średnie opóźnienie (min)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Rozkład opóźnień dla lotów polskich vs międzynarodowych
    polish_delays = polish_flights['delay_minutes']
    intl_delays = df[~((df['country_origin'] == 'Polska') | 
                      (df['country_destination'] == 'Polska'))]['delay_minutes']
    
    ax3.hist([polish_delays, intl_delays], bins=30, alpha=0.7, 
             label=['Loty polskie', 'Loty międzynarodowe'], color=['blue', 'orange'])
    ax3.set_title('Rozkład opóźnień: Polska vs międzynarodowe')
    ax3.set_xlabel('Opóźnienie (min)')
    ax3.set_ylabel('Liczba lotów')
    ax3.legend()
    ax3.set_xlim(0, 120)
    
    # Punktualność według miesięcy
    monthly_punctuality = polish_flights.groupby('month').apply(
        lambda x: (x['delay_minutes'] == 0).mean() * 100
    )
    monthly_punctuality.plot(kind='line', marker='o', ax=ax4, color='green', linewidth=2)
    ax4.set_title('Punktualność polskich lotów według miesięcy')
    ax4.set_xlabel('Miesiąc')
    ax4.set_ylabel('Punktualność (%)')
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(['Sty', 'Lut', 'Mar', 'Kwi', 'Maj', 'Cze',
                        'Lip', 'Sie', 'Wrz', 'Paź', 'Lis', 'Gru'])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return polish_flights, airport_df

def analyze_european_carriers(df):
    """Analiza europejskich przewoźników"""
    print("\n✈️  ANALIZA EUROPEJSKICH PRZEWOŹNIKÓW")
    print("="*60)
    
    # Statystyki przewoźników
    carrier_stats = df.groupby('airline').agg({
        'delay_minutes': ['count', 'mean', lambda x: (x == 0).mean() * 100],
        'distance_km': 'mean'
    }).round(1)
    
    carrier_stats.columns = ['Liczba_lotów', 'Średnie_opóźnienie', 'Punktualność_%', 'Średni_dystans']
    carrier_stats = carrier_stats.sort_values('Liczba_lotów', ascending=False)
    
    print("📈 Top 10 przewoźników według liczby lotów:")
    for i, (airline, row) in enumerate(carrier_stats.head(10).iterrows(), 1):
        print(f"   {i:2d}. {airline}")
        print(f"       • Lotów: {row['Liczba_lotów']:,}")
        print(f"       • Śr. opóźnienie: {row['Średnie_opóźnienie']:.1f} min")
        print(f"       • Punktualność: {row['Punktualność_%']:.1f}%")
        print(f"       • Śr. dystans: {row['Średni_dystans']:.0f} km")
    
    # Analiza LOT-u vs konkurencji
    if 'LOT Polish Airlines' in df['airline'].values:
        lot_stats = carrier_stats.loc['LOT Polish Airlines']
        other_carriers = carrier_stats[carrier_stats.index != 'LOT Polish Airlines']
        
        print(f"\n🇵🇱 LOT Polish Airlines vs konkurencja:")
        print(f"   • LOT - punktualność: {lot_stats['Punktualność_%']:.1f}%")
        print(f"   • Konkurencja - średnia punktualność: {other_carriers['Punktualność_%'].mean():.1f}%")
        print(f"   • LOT - średnie opóźnienie: {lot_stats['Średnie_opóźnienie']:.1f} min")
        print(f"   • Konkurencja - średnie opóźnienie: {other_carriers['Średnie_opóźnienie'].mean():.1f} min")
    
    # Wizualizacja
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Top przewoźnicy według liczby lotów
    top_carriers = carrier_stats.head(8)
    top_carriers['Liczba_lotów'].plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Top 8 przewoźników - liczba lotów')
    ax1.set_ylabel('Liczba lotów')
    ax1.tick_params(axis='x', rotation=45)
    
    # Punktualność vs liczba lotów
    ax2.scatter(carrier_stats['Liczba_lotów'], carrier_stats['Punktualność_%'], 
               alpha=0.7, s=100, color='orange')
    ax2.set_title('Punktualność vs liczba lotów')
    ax2.set_xlabel('Liczba lotów')
    ax2.set_ylabel('Punktualność (%)')
    ax2.grid(True, alpha=0.3)
    
    # Średnie opóźnienie według przewoźnika
    top_carriers['Średnie_opóźnienie'].plot(kind='bar', ax=ax3, color='lightcoral')
    ax3.set_title('Średnie opóźnienie - top przewoźnicy')
    ax3.set_ylabel('Średnie opóźnienie (min)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Rozkład typów opóźnień dla low-cost vs tradycyjnych
    low_cost = ['Ryanair', 'Wizz Air', 'easyJet', 'Norwegian']
    traditional = ['LOT Polish Airlines', 'Lufthansa', 'KLM', 'Air France', 'British Airways']
    
    low_cost_delays = df[df['airline'].isin(low_cost)]['delay_minutes']
    traditional_delays = df[df['airline'].isin(traditional)]['delay_minutes']
    
    ax4.hist([low_cost_delays, traditional_delays], bins=30, alpha=0.7,
             label=['Low-cost', 'Tradycyjne'], color=['red', 'blue'])
    ax4.set_title('Rozkład opóźnień: Low-cost vs Tradycyjne')
    ax4.set_xlabel('Opóźnienie (min)')
    ax4.set_ylabel('Liczba lotów')
    ax4.legend()
    ax4.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.show()
    
    return carrier_stats

def analyze_european_weather_patterns(df):
    """Analiza wzorców pogodowych w Europie"""
    print("\n🌦️  EUROPEJSKIE WZORCE POGODOWE")
    print("="*60)
    
    weather_delays = df[df['delay_reason'] == 'Weather'].copy()
    
    if len(weather_delays) == 0:
        print("❌ Brak danych o opóźnieniach pogodowych")
        return
    
    # Analiza sezonowa
    seasonal_weather = weather_delays.groupby('month').agg({
        'delay_minutes': ['count', 'mean']
    }).round(1)
    seasonal_weather.columns = ['Liczba_opóźnień', 'Średnie_opóźnienie']
    
    print("❄️ Sezonowość opóźnień pogodowych:")
    seasons = {
        'Zima (Gru-Lut)': [12, 1, 2],
        'Wiosna (Mar-Maj)': [3, 4, 5], 
        'Lato (Cze-Sie)': [6, 7, 8],
        'Jesień (Wrz-Lis)': [9, 10, 11]
    }
    
    for season, months in seasons.items():
        season_data = seasonal_weather.loc[seasonal_weather.index.isin(months)]
        total_delays = season_data['Liczba_opóźnień'].sum()
        avg_delay = season_data['Średnie_opóźnienie'].mean()
        print(f"   • {season}: {total_delays:,} opóźnień, średnio {avg_delay:.1f} min")
    
    # Najbardziej problematyczne regiony
    regional_weather = weather_delays.groupby('country_origin').agg({
        'delay_minutes': ['count', 'mean']
    }).round(1)
    regional_weather.columns = ['Liczba_opóźnień', 'Średnie_opóźnienie']
    regional_weather = regional_weather.sort_values('Liczba_opóźnień', ascending=False)
    
    print(f"\n🌍 Problematyczne regiony pogodowo:")
    for country, row in regional_weather.head(8).iterrows():
        print(f"   • {country}: {row['Liczba_opóźnień']:,} opóźnień, "
              f"średnio {row['Średnie_opóźnienie']:.1f} min")
    
    # Porównanie Polska vs Europa
    if 'Polska' in regional_weather.index:
        poland_weather = regional_weather.loc['Polska']
        europe_avg = regional_weather[regional_weather.index != 'Polska']['Średnie_opóźnienie'].mean()
        
        print(f"\n🇵🇱 Polska vs Europa - opóźnienia pogodowe:")
        print(f"   • Polska: {poland_weather['Średnie_opóźnienie']:.1f} min średnio")
        print(f"   • Europa (średnia): {europe_avg:.1f} min średnio")
        print(f"   • Różnica: {poland_weather['Średnie_opóźnienie'] - europe_avg:+.1f} min")
    
    return seasonal_weather, regional_weather

def analyze_strike_delays(df):
    """Analiza opóźnień związanych ze strajkami (specyfika europejska)"""
    print("\n✊ ANALIZA OPÓŹNIEŃ - STRAJKI (SPECYFIKA EUROPEJSKA)")
    print("="*60)
    
    strike_delays = df[df['delay_reason'] == 'Strike'].copy()
    
    if len(strike_delays) == 0:
        print("❌ Brak danych o opóźnieniach związanych ze strajkami")
        return
    
    print(f"📊 Ogólne statystyki strajków:")
    print(f"   • Liczba opóźnień: {len(strike_delays):,}")
    print(f"   • Procent wszystkich opóźnień: {len(strike_delays)/len(df[df['delay_minutes'] > 0])*100:.1f}%")
    print(f"   • Średnie opóźnienie: {strike_delays['delay_minutes'].mean():.1f} min")
    
    # Najbardziej dotknięte kraje
    country_strikes = strike_delays.groupby('country_origin').agg({
        'delay_minutes': ['count', 'mean']
    }).round(1)
    country_strikes.columns = ['Liczba_strajków', 'Średnie_opóźnienie']
    country_strikes = country_strikes.sort_values('Liczba_strajków', ascending=False)
    
    print(f"\n🌍 Kraje najbardziej dotknięte strajkami:")
    for country, row in country_strikes.head(6).iterrows():
        print(f"   • {country}: {row['Liczba_strajków']:,} przypadków, "
              f"średnio {row['Średnie_opóźnienie']:.1f} min")
    
    # Przewoźnicy a strajki
    airline_strikes = strike_delays.groupby('airline').size().sort_values(ascending=False)
    print(f"\n✈️ Przewoźnicy najbardziej dotknięci strajkami:")
    for airline, count in airline_strikes.head(5).items():
        print(f"   • {airline}: {count:,} opóźnień")
    
    return strike_delays, country_strikes

def create_european_summary_dashboard(df):
    """Tworzy podsumowanie analizy europejskiej"""
    print("\n📋 PODSUMOWANIE - EUROPEJSKI RUCH LOTNICZY")
    print("="*70)
    
    # Ogólne statystyki
    total_flights = len(df)
    avg_delay = df['delay_minutes'].mean()
    punctuality = (df['delay_minutes'] == 0).mean() * 100
    avg_distance = df['distance_km'].mean()
    
    print(f"🎯 KLUCZOWE METRYKI:")
    print(f"   • Łączna liczba lotów: {total_flights:,}")
    print(f"   • Średnie opóźnienie: {avg_delay:.1f} min")
    print(f"   • Punktualność: {punctuality:.1f}%")
    print(f"   • Średni dystans: {avg_distance:.0f} km")
    
    # Top kraje
    country_stats = df.groupby('country_origin').agg({
        'delay_minutes': ['count', 'mean', lambda x: (x == 0).mean() * 100]
    }).round(1)
    country_stats.columns = ['Liczba_lotów', 'Średnie_opóźnienie', 'Punktualność_%']
    country_stats = country_stats.sort_values('Liczba_lotów', ascending=False)
    
    print(f"\n🌍 TOP 8 KRAJÓW WEDŁUG RUCHU:")
    for i, (country, row) in enumerate(country_stats.head(8).iterrows(), 1):
        print(f"   {i}. {country}: {row['Liczba_lotów']:,} lotów, "
              f"punktualność {row['Punktualność_%']:.1f}%")
    
    # Przyczyny opóźnień
    delay_reasons = df[df['delay_minutes'] > 0]['delay_reason'].value_counts()
    delay_reasons_pct = (delay_reasons / delay_reasons.sum() * 100).round(1)
    
    print(f"\n⏰ PRZYCZYNY OPÓŹNIEŃ:")
    for reason, count in delay_reasons.items():
        pct = delay_reasons_pct[reason]
        print(f"   • {reason}: {count:,} ({pct}%)")
    
    # Rekomendacje
    print(f"\n💡 KLUCZOWE WNIOSKI I REKOMENDACJE:")
    
    # Najlepsze/najgorsze lotniska
    airport_performance = df.groupby('origin').agg({
        'delay_minutes': lambda x: (x == 0).mean() * 100
    }).round(1)
    airport_performance.columns = ['Punktualność_%']
    
    best_airports = airport_performance.nlargest(3, 'Punktualność_%')
    worst_airports = airport_performance.nsmallest(3, 'Punktualność_%')
    
    print(f"   ✅ Najbardziej punktualne lotniska:")
    for airport, perf in best_airports.iterrows():
        print(f"      • {airport}: {perf['Punktualność_%']:.1f}% punktualności")
    
    print(f"   ❌ Lotniska wymagające uwagi:")
    for airport, perf in worst_airports.iterrows():
        print(f"      • {airport}: {perf['Punktualność_%']:.1f}% punktualności")
    
    # Sezonowe rekomendacje
    seasonal_punctuality = df.groupby('month').apply(
        lambda x: (x['delay_minutes'] == 0).mean() * 100
    ).round(1)
    
    best_months = seasonal_punctuality.nlargest(3)
    worst_months = seasonal_punctuality.nsmallest(3)
    
    print(f"   📅 Najlepsze miesiące do podróży:")
    months_names = ['', 'Styczeń', 'Luty', 'Marzec', 'Kwiecień', 'Maj', 'Czerwiec',
                   'Lipiec', 'Sierpień', 'Wrzesień', 'Październik', 'Listopad', 'Grudzień']
    for month, punct in best_months.items():
        print(f"      • {months_names[month]}: {punct:.1f}% punktualności")
    
    print(f"   ⚠️  Miesiące z większym ryzykiem opóźnień:")
    for month, punct in worst_months.items():
        print(f"      • {months_names[month]}: {punct:.1f}% punktualności")
    
    return {
        'country_stats': country_stats,
        'delay_reasons': delay_reasons,
        'airport_performance': airport_performance,
        'seasonal_punctuality': seasonal_punctuality
    }

def run_complete_european_analysis(df):
    """Uruchamia kompletną analizę europejską"""
    print("🇪🇺 ROZPOCZYNANIE KOMPLETNEJ ANALIZY EUROPEJSKIEJ")
    print("="*70)
    
    results = {}
    
    # Analiza tras polskich
    results['polish_analysis'] = analyze_polish_routes(df)
    
    # Analiza przewoźników
    results['carrier_analysis'] = analyze_european_carriers(df)
    
    # Analiza wzorców pogodowych
    results['weather_analysis'] = analyze_european_weather_patterns(df)
    
    # Analiza strajków
    results['strike_analysis'] = analyze_strike_delays(df)
    
    # Podsumowanie
    results['summary'] = create_european_summary_dashboard(df)
    
    print("\n✅ ANALIZA EUROPEJSKA ZAKOŃCZONA")
    print("="*70)
    
    return results 