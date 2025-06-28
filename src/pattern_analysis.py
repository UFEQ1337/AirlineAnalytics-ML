"""
Moduł do zaawansowanej analizy wzorców opóźnień lotniczych.
Zawiera funkcje do analizy przyczyn, wzorców czasowych, geograficznych i samolotów.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
import warnings
warnings.filterwarnings('ignore')

def analyze_delay_reasons(df):
    """
    Kompleksowa analiza przyczyn opóźnień
    """
    print("🔍 ANALIZA PRZYCZYN OPÓŹNIEŃ")
    print("="*50)
    
    # Rozkład przyczyn opóźnień
    delay_reasons = df[df['delay_minutes'] > 0]['delay_reason'].value_counts()
    delay_reasons_pct = (delay_reasons / delay_reasons.sum() * 100).round(1)
    
    print("📊 Rozkład przyczyn opóźnień:")
    for reason, count in delay_reasons.items():
        pct = delay_reasons_pct[reason]
        print(f"   • {reason}: {count:,} ({pct}%)")
    
    # Średnie opóźnienie dla każdej przyczyny
    avg_delays = df[df['delay_minutes'] > 0].groupby('delay_reason')['delay_minutes'].agg(['mean', 'median', 'std']).round(1)
    print(f"\n⏰ Średnie opóźnienia według przyczyn:")
    for reason in avg_delays.index:
        mean_delay = avg_delays.loc[reason, 'mean']
        median_delay = avg_delays.loc[reason, 'median']
        print(f"   • {reason}: średnia {mean_delay} min, mediana {median_delay} min")
    
    # Wykres kołowy
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart
    colors = plt.get_cmap('Set3')(np.linspace(0, 1, len(delay_reasons)))
    wedges, texts, autotexts = ax1.pie(delay_reasons.values, labels=delay_reasons.index, 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Rozkład przyczyn opóźnień', fontsize=14, fontweight='bold')
    
    # Bar chart z średnimi opóźnieniami
    avg_delays['mean'].plot(kind='bar', ax=ax2, color=colors[:len(avg_delays)])
    ax2.set_title('Średnie opóźnienie według przyczyny', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Średnie opóźnienie (minuty)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return delay_reasons, avg_delays

def analyze_weather_delays(df):
    """
    Szczegółowa analiza opóźnień pogodowych
    """
    print("\n🌦️  ANALIZA OPÓŹNIEŃ POGODOWYCH")
    print("="*50)
    
    weather_delays = df[df['delay_reason'] == 'Weather'].copy()
    
    if len(weather_delays) == 0:
        print("❌ Brak danych o opóźnieniach pogodowych")
        return
    
    # Analiza miesięczna
    monthly_weather = weather_delays.groupby('month').agg({
        'delay_minutes': ['count', 'mean']
    }).round(1)
    monthly_weather.columns = ['Liczba_opóźnień', 'Średnie_opóźnienie']
    
    print("📅 Opóźnienia pogodowe według miesięcy:")
    months = ['Sty', 'Lut', 'Mar', 'Kwi', 'Maj', 'Cze', 
              'Lip', 'Sie', 'Wrz', 'Paź', 'Lis', 'Gru']
    for month_num in sorted(monthly_weather.index):
        count = monthly_weather.loc[month_num, 'Liczba_opóźnień']
        avg = monthly_weather.loc[month_num, 'Średnie_opóźnienie']
        print(f"   • {months[month_num-1]}: {count:,} opóźnień, średnio {avg} min")
    
    # Najgorsze lotniska
    airport_weather = weather_delays.groupby('origin').agg({
        'delay_minutes': ['count', 'mean']
    }).round(1)
    airport_weather.columns = ['Liczba_opóźnień', 'Średnie_opóźnienie']
    top_airports = airport_weather.nlargest(10, 'Liczba_opóźnień')
    
    print(f"\n🛫 Top 10 lotnisk z opóźnieniami pogodowymi:")
    for airport in top_airports.index:
        count = top_airports.loc[airport, 'Liczba_opóźnień']
        avg = top_airports.loc[airport, 'Średnie_opóźnienie']
        print(f"   • {airport}: {count:,} opóźnień, średnio {avg} min")
    
    # Wizualizacja
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Miesięczny trend
    monthly_weather['Liczba_opóźnień'].plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Liczba opóźnień pogodowych według miesięcy')
    ax1.set_xlabel('Miesiąc')
    ax1.set_ylabel('Liczba opóźnień')
    ax1.set_xticklabels([months[i-1] for i in monthly_weather.index], rotation=45)
    
    # Top lotniska
    top_airports['Liczba_opóźnień'].plot(kind='bar', ax=ax2, color='orange')
    ax2.set_title('Top 10 lotnisk - opóźnienia pogodowe')
    ax2.set_xlabel('Lotnisko')
    ax2.set_ylabel('Liczba opóźnień')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return monthly_weather, top_airports

def temporal_heatmap(df):
    """
    Tworzy heatmapę wzorców czasowych
    """
    print("\n🕒 ANALIZA WZORCÓW CZASOWYCH - HEATMAPA")
    print("="*50)
    
    # Przygotowanie danych
    df_temp = df.copy()
    df_temp['day_name'] = df_temp['day_of_week'].map({
        0: 'Pon', 1: 'Wt', 2: 'Śr', 3: 'Czw', 4: 'Pt', 5: 'Sob', 6: 'Nie'
    })
    
    # Heatmapa: godzina vs dzień tygodnia
    heatmap_data = df_temp.groupby(['hour', 'day_name'])['delay_minutes'].mean().unstack()
    heatmap_data = heatmap_data.reindex(columns=['Pon', 'Wt', 'Śr', 'Czw', 'Pt', 'Sob', 'Nie'])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Średnie opóźnienie (minuty)'})
    plt.title('Heatmapa: Średnie opóźnienie według godziny i dnia tygodnia')
    plt.xlabel('Dzień tygodnia')
    plt.ylabel('Godzina')
    plt.tight_layout()
    plt.show()
    
    # Rush hours analysis
    rush_morning = df_temp[df_temp['hour'].isin([6, 7, 8, 9])]
    rush_evening = df_temp[df_temp['hour'].isin([17, 18, 19, 20])]
    normal_hours = df_temp[~df_temp['hour'].isin([6, 7, 8, 9, 17, 18, 19, 20])]
    
    print("🚀 ANALIZA RUSH HOURS:")
    print(f"   • Poranne rush hours (6-9): średnie opóźnienie {rush_morning['delay_minutes'].mean():.1f} min")
    print(f"   • Wieczorne rush hours (17-20): średnie opóźnienie {rush_evening['delay_minutes'].mean():.1f} min")
    print(f"   • Pozostałe godziny: średnie opóźnienie {normal_hours['delay_minutes'].mean():.1f} min")
    
    # Late night analysis
    late_night = df_temp[df_temp['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5])]
    print(f"   • Loty nocne (22-5): średnie opóźnienie {late_night['delay_minutes'].mean():.1f} min")
    
    return heatmap_data

def cascading_delays_analysis(df):
    """
    Analiza efektu kaskadowego opóźnień
    """
    print("\n⛓️  ANALIZA EFEKTU KASKADOWEGO")
    print("="*50)
    
    # Grupowanie według godziny
    hourly_delays = df.groupby('hour')['delay_minutes'].agg(['mean', 'count']).round(1)
    
    # Pierwsze loty vs późniejsze
    early_flights = df[df['hour'] <= 8]
    late_flights = df[df['hour'] >= 18]
    mid_flights = df[(df['hour'] > 8) & (df['hour'] < 18)]
    
    print("📊 Efekt kaskadowy w ciągu dnia:")
    print(f"   • Wczesne loty (≤8): średnie opóźnienie {early_flights['delay_minutes'].mean():.1f} min")
    print(f"   • Loty środkowe (9-17): średnie opóźnienie {mid_flights['delay_minutes'].mean():.1f} min")
    print(f"   • Późne loty (≥18): średnie opóźnienie {late_flights['delay_minutes'].mean():.1f} min")
    
    # Weekend vs weekday
    weekend_delays = df[df['is_weekend']]['delay_minutes'].mean()
    weekday_delays = df[~df['is_weekend']]['delay_minutes'].mean()
    
    print(f"\n📅 Weekend vs Dzień roboczy:")
    print(f"   • Weekend: średnie opóźnienie {weekend_delays:.1f} min")
    print(f"   • Dni robocze: średnie opóźnienie {weekday_delays:.1f} min")
    
    # Wizualizacja
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Trend godzinowy
    hourly_delays['mean'].plot(kind='line', marker='o', ax=ax1, color='red', linewidth=2)
    ax1.set_title('Średnie opóźnienie według godziny')
    ax1.set_xlabel('Godzina')
    ax1.set_ylabel('Średnie opóźnienie (minuty)')
    ax1.grid(True, alpha=0.3)
    
    # Weekend vs weekday
    weekend_comparison = pd.Series({
        'Weekend': weekend_delays,
        'Dni robocze': weekday_delays
    })
    weekend_comparison.plot(kind='bar', ax=ax2, color=['skyblue', 'orange'])
    ax2.set_title('Porównanie: Weekend vs Dni robocze')
    ax2.set_ylabel('Średnie opóźnienie (minuty)')
    ax2.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    return hourly_delays

def geographic_patterns(df):
    """
    Analiza wzorców geograficznych
    """
    print("\n🗺️  ANALIZA WZORCÓW GEOGRAFICZNYCH")
    print("="*50)
    
    # Top 20 tras z największymi opóźnieniami
    df['route'] = df['origin'] + ' → ' + df['destination']
    route_analysis = df.groupby('route').agg({
        'delay_minutes': ['mean', 'count'],
        'distance_miles': 'first'
    }).round(1)
    route_analysis.columns = ['Średnie_opóźnienie', 'Liczba_lotów', 'Dystans']
    
    # Filtruj trasy z minimum 100 lotami
    frequent_routes = route_analysis[route_analysis['Liczba_lotów'] >= 100]
    top_delayed_routes = frequent_routes.nlargest(20, 'Średnie_opóźnienie')
    
    print("🛫 Top 20 tras z największymi opóźnieniami:")
    for i, (route, data) in enumerate(top_delayed_routes.iterrows(), 1):
        avg_delay = data['Średnie_opóźnienie']
        flights = data['Liczba_lotów']
        distance = data['Dystans']
        print(f"   {i:2d}. {route}: {avg_delay} min średnio ({flights:,} lotów, {distance} mil)")
    
    # Korelacja dystans vs opóźnienie
    correlation = df['distance_miles'].corr(df['delay_minutes'])
    print(f"\n📏 Korelacja dystans vs opóźnienie: {correlation:.3f}")
    
    # Hub airports vs regional airports
    flight_counts = df['origin'].value_counts()
    hub_airports = flight_counts.head(10).index.tolist()  # Top 10 jako hub airports
    
    hub_delays = df[df['origin'].isin(hub_airports)]['delay_minutes'].mean()
    regional_delays = df[~df['origin'].isin(hub_airports)]['delay_minutes'].mean()
    
    print(f"\n🏢 Hub airports vs Regional airports:")
    print(f"   • Hub airports (Top 10): średnie opóźnienie {hub_delays:.1f} min")
    print(f"   • Regional airports: średnie opóźnienie {regional_delays:.1f} min")
    
    # Wizualizacja
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top 15 tras
    top_15_routes = top_delayed_routes.head(15)
    top_15_routes['Średnie_opóźnienie'].plot(kind='barh', ax=ax1, color='coral')
    ax1.set_title('Top 15 tras z największymi opóźnieniami')
    ax1.set_xlabel('Średnie opóźnienie (minuty)')
    
    # Scatter plot dystans vs opóźnienie
    sample_data = df.sample(min(5000, len(df)))  # Sample dla lepszej czytelności
    ax2.scatter(sample_data['distance_miles'], sample_data['delay_minutes'], 
               alpha=0.5, s=20, color='blue')
    ax2.set_xlabel('Dystans (mile)')
    ax2.set_ylabel('Opóźnienie (minuty)')
    ax2.set_title(f'Dystans vs Opóźnienie (r = {correlation:.3f})')
    
    plt.tight_layout()
    plt.show()
    
    return top_delayed_routes, correlation

def aircraft_performance(df):
    """
    Analiza wydajności samolotów
    """
    print("\n✈️  ANALIZA WYDAJNOŚCI SAMOLOTÓW")
    print("="*50)
    
    # Dodanie symulowanego wieku samolotu
    np.random.seed(42)  # dla reprodukowalności
    df_aircraft = df.copy()
    
    # Symulacja wieku samolotu (0-30 lat)
    aircraft_ages = {}
    for aircraft_type in df_aircraft['aircraft_type'].unique():
        if 'Boeing 737' in aircraft_type:
            avg_age = 12  # Starsze samoloty
        elif 'Boeing 787' in aircraft_type or 'Airbus A350' in aircraft_type:
            avg_age = 5   # Nowsze samoloty
        elif 'Embraer' in aircraft_type or 'CRJ' in aircraft_type:
            avg_age = 8   # Średnio stare
        else:
            avg_age = 10  # Średnia
        
        # Generuj wiek z rozkładem normalnym
        ages = np.random.normal(avg_age, 3, size=1000)
        ages = np.clip(ages, 0, 30)
        aircraft_ages[aircraft_type] = ages
    
    # Przypisz wiek do każdego lotu
    df_aircraft['aircraft_age'] = df_aircraft['aircraft_type'].map(
        lambda x: np.random.choice(aircraft_ages[x])
    ).round(1)
    
    # Analiza według typu samolotu
    aircraft_analysis = df_aircraft.groupby('aircraft_type').agg({
        'delay_minutes': ['mean', 'count'],
        'aircraft_age': 'mean'
    }).round(1)
    aircraft_analysis.columns = ['Średnie_opóźnienie', 'Liczba_lotów', 'Średni_wiek']
    
    # Sortuj według opóźnień
    aircraft_sorted = aircraft_analysis.sort_values('Średnie_opóźnienie', ascending=False)
    
    print("🛩️  Wydajność według typu samolotu:")
    for aircraft, data in aircraft_sorted.iterrows():
        avg_delay = data['Średnie_opóźnienie']
        flights = data['Liczba_lotów']
        avg_age = data['Średni_wiek']
        print(f"   • {aircraft}: {avg_delay} min średnio ({flights:,} lotów, śr. wiek {avg_age} lat)")
    
    # Korelacja wiek vs opóźnienie
    age_correlation = df_aircraft['aircraft_age'].corr(df_aircraft['delay_minutes'])
    print(f"\n📈 Korelacja wiek samolotu vs opóźnienie: {age_correlation:.3f}")
    
    # Maintenance patterns (symulacja)
    print(f"\n🔧 Wzorce konserwacji (symulacja):")
    old_aircraft = df_aircraft[df_aircraft['aircraft_age'] > 15]
    new_aircraft = df_aircraft[df_aircraft['aircraft_age'] <= 5]
    
    old_mechanical = (old_aircraft['delay_reason'] == 'Mechanical').mean() * 100
    new_mechanical = (new_aircraft['delay_reason'] == 'Mechanical').mean() * 100
    
    print(f"   • Stare samoloty (>15 lat): {old_mechanical:.1f}% opóźnień mechanicznych")
    print(f"   • Nowe samoloty (≤5 lat): {new_mechanical:.1f}% opóźnień mechanicznych")
    
    # Wizualizacja
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top/Bottom 10 typów samolotów
    top_bottom = pd.concat([aircraft_sorted.head(5), aircraft_sorted.tail(5)])
    colors = ['red'] * 5 + ['green'] * 5
    top_bottom['Średnie_opóźnienie'].plot(kind='barh', ax=ax1, color=colors)
    ax1.set_title('Najgorsze i najlepsze typy samolotów')
    ax1.set_xlabel('Średnie opóźnienie (minuty)')
    
    # Wiek vs opóźnienie
    sample_aircraft = df_aircraft.sample(min(3000, len(df_aircraft)))
    ax2.scatter(sample_aircraft['aircraft_age'], sample_aircraft['delay_minutes'], 
               alpha=0.5, s=20, color='purple')
    ax2.set_xlabel('Wiek samolotu (lata)')
    ax2.set_ylabel('Opóźnienie (minuty)')
    ax2.set_title(f'Wiek samolotu vs Opóźnienie (r = {age_correlation:.3f})')
    
    plt.tight_layout()
    plt.show()
    
    return aircraft_analysis, df_aircraft

def create_interactive_map(df):
    """
    Tworzy interaktywną mapę z plotly
    """
    print("\n🗺️  TWORZENIE INTERAKTYWNEJ MAPY")
    print("="*50)
    
    # Współrzędne lotnisk (przykładowe)
    airport_coords = {
        'JFK': {'lat': 40.6413, 'lon': -73.7781, 'city': 'New York'},
        'LAX': {'lat': 33.9425, 'lon': -118.4081, 'city': 'Los Angeles'},
        'ORD': {'lat': 41.9742, 'lon': -87.9073, 'city': 'Chicago'},
        'DFW': {'lat': 32.8998, 'lon': -97.0403, 'city': 'Dallas'},
        'DEN': {'lat': 39.8561, 'lon': -104.6737, 'city': 'Denver'},
        'SFO': {'lat': 37.6213, 'lon': -122.3790, 'city': 'San Francisco'},
        'SEA': {'lat': 47.4502, 'lon': -122.3088, 'city': 'Seattle'},
        'LAS': {'lat': 36.0840, 'lon': -115.1537, 'city': 'Las Vegas'},
        'PHX': {'lat': 33.4484, 'lon': -112.0740, 'city': 'Phoenix'},
        'IAH': {'lat': 29.9902, 'lon': -95.3368, 'city': 'Houston'},
        'CLT': {'lat': 35.2144, 'lon': -80.9473, 'city': 'Charlotte'},
        'MIA': {'lat': 25.7932, 'lon': -80.2906, 'city': 'Miami'},
        'BOS': {'lat': 42.3656, 'lon': -71.0096, 'city': 'Boston'},
        'MSP': {'lat': 44.8848, 'lon': -93.2223, 'city': 'Minneapolis'},
        'FLL': {'lat': 26.0742, 'lon': -80.1506, 'city': 'Fort Lauderdale'},
        'DTW': {'lat': 42.2162, 'lon': -83.3554, 'city': 'Detroit'},
        'PHL': {'lat': 39.8744, 'lon': -75.2424, 'city': 'Philadelphia'},
        'LGA': {'lat': 40.7769, 'lon': -73.8740, 'city': 'New York LGA'},
        'BWI': {'lat': 39.1774, 'lon': -76.6684, 'city': 'Baltimore'},
        'MDW': {'lat': 41.7868, 'lon': -87.7505, 'city': 'Chicago Midway'}
    }
    
    # Agregacja danych dla każdego lotniska
    airport_stats = df.groupby('origin').agg({
        'delay_minutes': 'mean',
        'flight_date': 'count'
    }).round(1)
    airport_stats.columns = ['avg_delay', 'flight_count']
    
    # Przygotowanie danych do mapy
    map_data = []
    for airport in airport_stats.index:
        if airport in airport_coords:
            coord = airport_coords[airport]
            stats = airport_stats.loc[airport]
            
            map_data.append({
                'airport': airport,
                'city': coord['city'],
                'lat': coord['lat'],
                'lon': coord['lon'],
                'avg_delay': stats['avg_delay'],
                'flight_count': stats['flight_count']
            })
    
    map_df = pd.DataFrame(map_data)
    
    # Utworzenie interaktywnej mapy
    fig = px.scatter_mapbox(
        map_df,
        lat='lat',
        lon='lon',
        size='flight_count',
        color='avg_delay',
        hover_name='city',
        hover_data={
            'airport': True,
            'avg_delay': ':.1f',
            'flight_count': ':,',
            'lat': False,
            'lon': False
        },
        color_continuous_scale='RdYlBu_r',
        size_max=30,
        zoom=3,
        center={'lat': 39.8283, 'lon': -98.5795},  # Centrum USA
        title='Mapa lotnisk USA - Średnie opóźnienia i liczba lotów'
    )
    
    fig.update_layout(
        mapbox_style='open-street-map',
        height=600,
        margin={'r': 0, 't': 30, 'l': 0, 'b': 0}
    )
    
    fig.show()
    
    print("✅ Interaktywna mapa utworzona!")
    print("   • Rozmiar punktu = liczba lotów")
    print("   • Kolor = średnie opóźnienie")
    
    return map_df

def statistical_testing(df):
    """
    Przeprowadza testy statystyczne
    """
    print("\n📊 TESTY STATYSTYCZNE")
    print("="*60)
    
    # T-test: Weekend vs Dni robocze
    weekend_delays = df[df['is_weekend']]['delay_minutes']
    weekday_delays = df[~df['is_weekend']]['delay_minutes']
    
    t_stat, t_p_value = stats.ttest_ind(weekend_delays, weekday_delays)  # type: ignore
    
    print("1️⃣  T-test: Weekend vs Dni robocze")
    print(f"   • Średnia weekend: {weekend_delays.mean():.2f} min")
    print(f"   • Średnia dni robocze: {weekday_delays.mean():.2f} min")
    print(f"   • T-statistic: {t_stat:.4f}")
    print(f"   • P-value: {t_p_value:.6f}")
    if t_p_value < 0.05:  # type: ignore
        print("   ✅ Różnica jest statystycznie istotna (p < 0.05)")
    else:
        print("   ❌ Różnica nie jest statystycznie istotna (p ≥ 0.05)")
    
    # ANOVA: Różnice między liniami lotniczymi
    airlines = df['airline'].unique()
    airline_delays = [df[df['airline'] == airline]['delay_minutes'] for airline in airlines]
    
    f_stat, f_p_value = f_oneway(*airline_delays)  # type: ignore
    
    print(f"\n2️⃣  ANOVA: Różnice między liniami lotniczymi")
    print(f"   • F-statistic: {f_stat:.4f}")
    print(f"   • P-value: {f_p_value:.6f}")
    if f_p_value < 0.05:  # type: ignore
        print("   ✅ Istnieją statystycznie istotne różnice między liniami (p < 0.05)")
    else:
        print("   ❌ Brak statystycznie istotnych różnic między liniami (p ≥ 0.05)")
    
    # Chi-square: Niezależność przyczyn opóźnień od pory roku
    # Podziel rok na pory roku
    df_seasonal = df.copy()
    df_seasonal['season'] = df_seasonal['month'].map({
        12: 'Zima', 1: 'Zima', 2: 'Zima',
        3: 'Wiosna', 4: 'Wiosna', 5: 'Wiosna',
        6: 'Lato', 7: 'Lato', 8: 'Lato',
        9: 'Jesień', 10: 'Jesień', 11: 'Jesień'
    })
    
    # Tabela kontyngencji
    delayed_flights = df_seasonal[df_seasonal['delay_minutes'] > 0]
    contingency_table = pd.crosstab(delayed_flights['season'], delayed_flights['delay_reason'])
    
    chi2_stat, chi2_p_value, dof, expected = chi2_contingency(contingency_table)  # type: ignore
    
    print(f"\n3️⃣  Chi-square: Przyczyny opóźnień vs Pora roku")
    print(f"   • Chi-square statistic: {chi2_stat:.4f}")
    print(f"   • P-value: {chi2_p_value:.6f}")
    print(f"   • Degrees of freedom: {dof}")
    if chi2_p_value < 0.05:  # type: ignore
        print("   ✅ Przyczyny opóźnień zależą od pory roku (p < 0.05)")
    else:
        print("   ❌ Przyczyny opóźnień nie zależą od pory roku (p ≥ 0.05)")
    
    print(f"\n📋 Tabela kontyngencji:")
    print(contingency_table)
    
    return {
        'weekend_ttest': (t_stat, t_p_value),
        'airline_anova': (f_stat, f_p_value),
        'seasonal_chi2': (chi2_stat, chi2_p_value, dof),
        'contingency_table': contingency_table
    }

def find_key_insights(df):
    """
    Znajdź kluczowe wnioski z analizy
    """
    print("\n🔑 KLUCZOWE WNIOSKI")
    print("="*60)
    
    # 1. Najgorsze kombinacje: dzień + godzina + linia
    df_insights = df.copy()
    df_insights['day_name'] = df_insights['day_of_week'].map({
        0: 'Poniedziałek', 1: 'Wtorek', 2: 'Środa', 3: 'Czwartek', 
        4: 'Piątek', 5: 'Sobota', 6: 'Niedziela'
    })
    
    worst_combinations = df_insights.groupby(['day_name', 'hour', 'airline'])['delay_minutes'].agg(['mean', 'count']).reset_index()
    worst_combinations = worst_combinations[worst_combinations['count'] >= 20]  # Min 20 lotów
    worst_combinations = worst_combinations.nlargest(10, 'mean')
    
    print("1️⃣  Najgorsze kombinacje (dzień + godzina + linia):")
    for i, row in worst_combinations.iterrows():
        day = row['day_name']
        hour = row['hour']
        airline = row['airline']
        avg_delay = row['mean']
        count = row['count']
        print(f"   • {day} {hour:02d}:00, {airline}: {avg_delay:.1f} min średnio ({count} lotów)")
    
    # 2. Seasonal patterns
    seasonal_analysis = df_insights.groupby(['month', 'delay_reason'])['delay_minutes'].count().unstack(fill_value=0)
    
    print(f"\n2️⃣  Wzorce sezonowe w przyczynach opóźnień:")
    months = ['Sty', 'Lut', 'Mar', 'Kwi', 'Maj', 'Cze', 
              'Lip', 'Sie', 'Wrz', 'Paź', 'Lis', 'Gru']
    
    for reason in ['Weather', 'Air Traffic', 'Mechanical']:
        if reason in seasonal_analysis.columns:
            peak_month = seasonal_analysis[reason].idxmax()
            peak_count = seasonal_analysis[reason].max()
            print(f"   • {reason}: szczyt w {months[peak_month-1]} ({peak_count:,} opóźnień)")
    
    # 3. Lotniska z największym efektem kaskadowym
    cascading_effect = df_insights.groupby(['origin', 'hour'])['delay_minutes'].mean().reset_index()
    cascading_variance = cascading_effect.groupby('origin')['delay_minutes'].var().sort_values(ascending=False)
    
    print(f"\n3️⃣  Lotniska z największym efektem kaskadowym:")
    for i, (airport, variance) in enumerate(cascading_variance.head(5).items(), 1):
        print(f"   {i}. {airport}: wariancja opóźnień {variance:.1f}")
    
    # 4. Cost impact estimation
    print(f"\n4️⃣  Szacowanie kosztów opóźnień:")
    
    # Dodaj kolumnę cost_per_minute (symulacja)
    np.random.seed(42)
    cost_per_minute_base = {
        'American Airlines': 45, 'Delta Air Lines': 42, 'United Airlines': 44,
        'Southwest Airlines': 35, 'JetBlue Airways': 38, 'Alaska Airlines': 40,
        'Spirit Airlines': 30, 'Frontier Airlines': 32, 'Allegiant Air': 28, 
        'Hawaiian Airlines': 50
    }
    
    df_insights['cost_per_minute'] = df_insights['airline'].map(cost_per_minute_base)
    df_insights['total_cost'] = df_insights['delay_minutes'] * df_insights['cost_per_minute']
    
    total_cost = df_insights['total_cost'].sum()
    avg_cost_per_flight = df_insights['total_cost'].mean()
    
    print(f"   • Całkowity koszt opóźnień: ${total_cost:,.0f}")
    print(f"   • Średni koszt na lot: ${avg_cost_per_flight:.0f}")
    
    # Top costly airlines
    airline_costs = df_insights.groupby('airline')['total_cost'].sum().sort_values(ascending=False)
    print(f"   • Najkosztowniejsze linie:")
    for i, (airline, cost) in enumerate(airline_costs.head(5).items(), 1):
        print(f"     {i}. {airline}: ${cost:,.0f}")
    
    return {
        'worst_combinations': worst_combinations,
        'seasonal_patterns': seasonal_analysis,
        'cascading_airports': cascading_variance,
        'total_cost': total_cost,
        'avg_cost_per_flight': avg_cost_per_flight
    } 