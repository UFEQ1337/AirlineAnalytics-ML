"""
Moduł wizualizacyjny dla analizy danych lotniczych.
Zawiera funkcje do tworzenia wykresów i analiz statystycznych.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Ustawienie stylu
plt.style.use('default')
sns.set_palette("husl")

def setup_polish_plots():
    """Konfiguracja wykresów dla języka polskiego"""
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 14

def plot_delay_distribution(df, save_path=None):
    """
    Tworzy wizualizację rozkładu opóźnień lotów.
    
    Args:
        df: DataFrame z danymi lotów
        save_path: Ścieżka do zapisania wykresu (opcjonalne)
    """
    setup_polish_plots()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analiza Rozkładu Opóźnień Lotów', fontsize=16, fontweight='bold')
    
    # Histogram opóźnień
    axes[0, 0].hist(df['delay_minutes'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Rozkład Opóźnień (wszystkie loty)')
    axes[0, 0].set_xlabel('Opóźnienie (minuty)')
    axes[0, 0].set_ylabel('Liczba Lotów')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram dla opóźnień <= 120 minut (lepszy widok)
    delayed_data = df[df['delay_minutes'] <= 120]['delay_minutes']
    axes[0, 1].hist(delayed_data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Rozkład Opóźnień (≤ 120 minut)')
    axes[0, 1].set_xlabel('Opóźnienie (minuty)')
    axes[0, 1].set_ylabel('Liczba Lotów')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot
    axes[1, 0].boxplot(df['delay_minutes'], vert=True)
    axes[1, 0].set_title('Box Plot Opóźnień')
    axes[1, 0].set_ylabel('Opóźnienie (minuty)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Statystyki podstawowe
    stats_text = f"""
    Średnie opóźnienie: {df['delay_minutes'].mean():.1f} min
    Mediana: {df['delay_minutes'].median():.1f} min
    
    Percentyle:
    50%: {df['delay_minutes'].quantile(0.5):.1f} min
    90%: {df['delay_minutes'].quantile(0.9):.1f} min
    95%: {df['delay_minutes'].quantile(0.95):.1f} min
    99%: {df['delay_minutes'].quantile(0.99):.1f} min
    
    Procent punktualnych lotów: {(df['delay_minutes'] == 0).mean()*100:.1f}%
    """
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                   verticalalignment='top', fontsize=10, bbox=dict(boxstyle="round", facecolor='wheat'))
    axes[1, 1].set_title('Statystyki Opisowe')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_airline_performance(df, save_path=None):
    """
    Tworzy wizualizację wydajności linii lotniczych.
    
    Args:
        df: DataFrame z danymi lotów
        save_path: Ścieżka do zapisania wykresu (opcjonalne)
    """
    setup_polish_plots()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analiza Wydajności Linii Lotniczych', fontsize=16, fontweight='bold')
    
    # Średnie opóźnienie według linii
    airline_avg_delay = df.groupby('airline')['delay_minutes'].mean().sort_values(ascending=False)
    top_5_worst = airline_avg_delay.head()
    
    axes[0, 0].bar(range(len(top_5_worst)), top_5_worst.values, color='lightcoral')
    axes[0, 0].set_title('Top 5 Linii z Największymi Opóźnieniami')
    axes[0, 0].set_xlabel('Linie Lotnicze')
    axes[0, 0].set_ylabel('Średnie Opóźnienie (minuty)')
    axes[0, 0].set_xticks(range(len(top_5_worst)))
    axes[0, 0].set_xticklabels(top_5_worst.index, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Procent punktualnych lotów
    airline_punctuality = df.groupby('airline').apply(
        lambda x: (x['delay_minutes'] == 0).mean() * 100
    ).sort_values(ascending=False)
    
    axes[0, 1].bar(range(len(airline_punctuality)), airline_punctuality.values, color='lightgreen')
    axes[0, 1].set_title('Punktualność Linii Lotniczych (%)')
    axes[0, 1].set_xlabel('Linie Lotnicze')
    axes[0, 1].set_ylabel('Procent Punktualnych Lotów')
    axes[0, 1].set_xticks(range(len(airline_punctuality)))
    axes[0, 1].set_xticklabels(airline_punctuality.index, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot opóźnień według linii
    df_plot = df[df['airline'].isin(airline_avg_delay.head(8).index)]
    sns.boxplot(data=df_plot, x='airline', y='delay_minutes', ax=axes[1, 0])
    axes[1, 0].set_title('Rozkład Opóźnień - Top 8 Linii')
    axes[1, 0].set_xlabel('Linie Lotnicze')
    axes[1, 0].set_ylabel('Opóźnienie (minuty)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Liczba lotów według linii
    flight_counts = df['airline'].value_counts()
    axes[1, 1].bar(range(len(flight_counts)), flight_counts.values, color='skyblue')
    axes[1, 1].set_title('Liczba Lotów według Linii')
    axes[1, 1].set_xlabel('Linie Lotnicze')
    axes[1, 1].set_ylabel('Liczba Lotów')
    axes[1, 1].set_xticks(range(len(flight_counts)))
    axes[1, 1].set_xticklabels(flight_counts.index, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_airport_heatmap(df, save_path=None):
    """
    Tworzy analizę lotnisk i tras.
    
    Args:
        df: DataFrame z danymi lotów
        save_path: Ścieżka do zapisania wykresu (opcjonalne)
    """
    setup_polish_plots()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analiza Lotnisk i Tras', fontsize=16, fontweight='bold')
    
    # Najgorsze lotniska wylotu
    origin_delays = df.groupby('origin')['delay_minutes'].mean().sort_values(ascending=False).head(10)
    axes[0, 0].barh(range(len(origin_delays)), origin_delays.values, color='orange')
    axes[0, 0].set_title('Top 10 Najgorszych Lotnisk Wylotu')
    axes[0, 0].set_xlabel('Średnie Opóźnienie (minuty)')
    axes[0, 0].set_ylabel('Lotnisko Wylotu')
    axes[0, 0].set_yticks(range(len(origin_delays)))
    axes[0, 0].set_yticklabels(origin_delays.index)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Najgorsze lotniska przylotu
    dest_delays = df.groupby('destination')['delay_minutes'].mean().sort_values(ascending=False).head(10)
    axes[0, 1].barh(range(len(dest_delays)), dest_delays.values, color='red')
    axes[0, 1].set_title('Top 10 Najgorszych Lotnisk Przylotu')
    axes[0, 1].set_xlabel('Średnie Opóźnienie (minuty)')
    axes[0, 1].set_ylabel('Lotnisko Przylotu')
    axes[0, 1].set_yticks(range(len(dest_delays)))
    axes[0, 1].set_yticklabels(dest_delays.index)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top 10 tras z największymi opóźnieniami
    df['route'] = df['origin'] + ' → ' + df['destination']
    route_delays = df.groupby('route')['delay_minutes'].mean().sort_values(ascending=False).head(10)
    axes[1, 0].barh(range(len(route_delays)), route_delays.values, color='purple')
    axes[1, 0].set_title('Top 10 Tras z Największymi Opóźnieniami')
    axes[1, 0].set_xlabel('Średnie Opóźnienie (minuty)')
    axes[1, 0].set_ylabel('Trasa')
    axes[1, 0].set_yticks(range(len(route_delays)))
    axes[1, 0].set_yticklabels(route_delays.index, fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Heatmapa - najpopularniejsze kombinacje lotnisk
    top_origins = df['origin'].value_counts().head(8).index
    top_dests = df['destination'].value_counts().head(8).index
    
    heatmap_data = df[df['origin'].isin(top_origins) & df['destination'].isin(top_dests)]
    pivot_table = heatmap_data.pivot_table(values='delay_minutes', 
                                          index='origin', 
                                          columns='destination', 
                                          aggfunc='mean')
    
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='Reds', ax=axes[1, 1])
    axes[1, 1].set_title('Heatmapa Opóźnień: Origin vs Destination')
    axes[1, 1].set_xlabel('Lotnisko Przylotu')
    axes[1, 1].set_ylabel('Lotnisko Wylotu')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_temporal_patterns(df, save_path=None):
    """
    Tworzy analizę wzorców czasowych opóźnień.
    
    Args:
        df: DataFrame z danymi lotów
        save_path: Ścieżka do zapisania wykresu (opcjonalne)
    """
    setup_polish_plots()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analiza Wzorców Czasowych Opóźnień', fontsize=16, fontweight='bold')
    
    # Opóźnienia według godziny dnia
    hourly_delays = df.groupby('hour')['delay_minutes'].mean()
    axes[0, 0].plot(hourly_delays.index, hourly_delays.values, marker='o', linewidth=2, markersize=6)
    axes[0, 0].set_title('Średnie Opóźnienie według Godziny Dnia')
    axes[0, 0].set_xlabel('Godzina')
    axes[0, 0].set_ylabel('Średnie Opóźnienie (minuty)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(range(0, 24, 2))
    
    # Opóźnienia według dnia tygodnia
    day_names = ['Poniedziałek', 'Wtorek', 'Środa', 'Czwartek', 'Piątek', 'Sobota', 'Niedziela']
    daily_delays = df.groupby('day_of_week')['delay_minutes'].mean()
    axes[0, 1].bar(range(7), daily_delays.values, color='lightcoral')
    axes[0, 1].set_title('Średnie Opóźnienie według Dnia Tygodnia')
    axes[0, 1].set_xlabel('Dzień Tygodnia')
    axes[0, 1].set_ylabel('Średnie Opóźnienie (minuty)')
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(day_names, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Opóźnienia według miesiąca
    month_names = ['Sty', 'Lut', 'Mar', 'Kwi', 'Maj', 'Cze', 
                   'Lip', 'Sie', 'Wrz', 'Paź', 'Lis', 'Gru']
    monthly_delays = df.groupby('month')['delay_minutes'].mean()
    axes[1, 0].bar(range(1, 13), monthly_delays.values, color='skyblue')
    axes[1, 0].set_title('Sezonowość - Opóźnienia według Miesiąca')
    axes[1, 0].set_xlabel('Miesiąc')
    axes[1, 0].set_ylabel('Średnie Opóźnienie (minuty)')
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].set_xticklabels(month_names)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Weekend vs dni robocze
    weekend_comparison = df.groupby('is_weekend')['delay_minutes'].mean()
    weekend_labels = ['Dni robocze', 'Weekend']
    axes[1, 1].bar(range(2), weekend_comparison.values, color=['lightgreen', 'orange'])
    axes[1, 1].set_title('Opóźnienia: Dni Robocze vs Weekend')
    axes[1, 1].set_xlabel('Typ Dnia')
    axes[1, 1].set_ylabel('Średnie Opóźnienie (minuty)')
    axes[1, 1].set_xticks(range(2))
    axes[1, 1].set_xticklabels(weekend_labels)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_summary_stats(df):
    """
    Tworzy podsumowanie statystyczne danych.
    
    Args:
        df: DataFrame z danymi lotów
        
    Returns:
        dict: Słownik ze statystykami
    """
    stats = {
        'podstawowe_statystyki': {
            'total_flights': len(df),
            'średnie_opóźnienie': df['delay_minutes'].mean(),
            'mediana_opóźnienia': df['delay_minutes'].median(),
            'procent_punktualnych': (df['delay_minutes'] == 0).mean() * 100,
            'najdłuższe_opóźnienie': df['delay_minutes'].max(),
            'najkrótsza_podróż': df['distance_miles'].min(),
            'najdłuższa_podróż': df['distance_miles'].max()
        },
        
        'najlepsza_linia': {
            'nazwa': df.groupby('airline')['delay_minutes'].mean().idxmin(),
            'średnie_opóźnienie': df.groupby('airline')['delay_minutes'].mean().min()
        },
        
        'najgorsza_linia': {
            'nazwa': df.groupby('airline')['delay_minutes'].mean().idxmax(),
            'średnie_opóźnienie': df.groupby('airline')['delay_minutes'].mean().max()
        },
        
        'najlepsza_godzina': {
            'godzina': df.groupby('hour')['delay_minutes'].mean().idxmin(),
            'średnie_opóźnienie': df.groupby('hour')['delay_minutes'].mean().min()
        },
        
        'najgorsza_godzina': {
            'godzina': df.groupby('hour')['delay_minutes'].mean().idxmax(),
            'średnie_opóźnienie': df.groupby('hour')['delay_minutes'].mean().max()
        },
        
        'najgorszy_miesiąc': {
            'miesiąc': df.groupby('month')['delay_minutes'].mean().idxmax(),
            'średnie_opóźnienie': df.groupby('month')['delay_minutes'].mean().max()
        }
    }
    
    return stats

def print_summary_stats(stats):
    """Wyświetla podsumowanie statystyczne w czytelnej formie."""
    print("="*60)
    print("           PODSUMOWANIE ANALIZY LOTÓW")
    print("="*60)
    
    basic = stats['podstawowe_statystyki']
    print(f"\n📊 PODSTAWOWE STATYSTYKI:")
    print(f"   • Całkowita liczba lotów: {basic['total_flights']:,}")
    print(f"   • Średnie opóźnienie: {basic['średnie_opóźnienie']:.1f} minut")
    print(f"   • Mediana opóźnień: {basic['mediana_opóźnienia']:.1f} minut")
    print(f"   • Procent punktualnych lotów: {basic['procent_punktualnych']:.1f}%")
    print(f"   • Najdłuższe opóźnienie: {basic['najdłuższe_opóźnienie']:.0f} minut")
    
    print(f"\n✈️  NAJLEPSZA LINIA LOTNICZA:")
    print(f"   • {stats['najlepsza_linia']['nazwa']}")
    print(f"   • Średnie opóźnienie: {stats['najlepsza_linia']['średnie_opóźnienie']:.1f} minut")
    
    print(f"\n❌ NAJGORSZA LINIA LOTNICZA:")
    print(f"   • {stats['najgorsza_linia']['nazwa']}")
    print(f"   • Średnie opóźnienie: {stats['najgorsza_linia']['średnie_opóźnienie']:.1f} minut")
    
    print(f"\n🕐 NAJLEPSZA GODZINA DO LOTU:")
    print(f"   • Godzina {stats['najlepsza_godzina']['godzina']}:00")
    print(f"   • Średnie opóźnienie: {stats['najlepsza_godzina']['średnie_opóźnienie']:.1f} minut")
    
    print(f"\n⏰ NAJGORSZA GODZINA DO LOTU:")
    print(f"   • Godzina {stats['najgorsza_godzina']['godzina']}:00")
    print(f"   • Średnie opóźnienie: {stats['najgorsza_godzina']['średnie_opóźnienie']:.1f} minut")
    
    print(f"\n📅 NAJGORSZY MIESIĄC:")
    print(f"   • Miesiąc {stats['najgorszy_miesiąc']['miesiąc']}")
    print(f"   • Średnie opóźnienie: {stats['najgorszy_miesiąc']['średnie_opóźnienie']:.1f} minut")
    
    print("="*60) 