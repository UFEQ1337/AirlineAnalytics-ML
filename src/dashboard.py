"""
🚀 INTERAKTYWNY DASHBOARD - AIRLINE ANALYTICS ML
=============================================

Dashboard Streamlit z trzema stronami:
1. Overview - KPI, timeline, real-time simulator
2. Analytics - filtry, wykresy dynamiczne, narzędzie porównawcze
3. Predictor - form przewidywania, analiza podobnych lotów

Autor: AirlineAnalytics-ML Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, Any, Optional, List

# Dodaj src do ścieżki
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from models import DelayClassifier, DelayRegressor, predict_delay
    from utils import load_clean_data, generate_report, model_health_check
    # Import funkcji visualization i pattern_analysis jeśli potrzebne
    import pattern_analysis
    import visualization
except ImportError as e:
    st.error(f"❌ Błąd importu modułów: {str(e)}")
    st.error("Upewnij się, że wszystkie pliki src/ są dostępne.")
    st.stop()

# ===========================
# KONFIGURACJA STRONY
# ===========================
st.set_page_config(
    page_title="✈️ Airline Analytics ML Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS dla lepszego wyglądu
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
    }
    .kpi-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f4e79;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: -5px;
    }
    .success-metric {
        color: #28a745;
    }
    .warning-metric {
        color: #ffc107;
    }
    .danger-metric {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# FUNKCJE POMOCNICZE
# ===========================

@st.cache_data(ttl=3600)  # Cache na 1 godzinę
def load_data():
    """Ładuje dane z cache"""
    try:
        if hasattr(st.session_state, 'clean_data'):
            return st.session_state.clean_data
        
        # Próbuj załadować z utils
        try:
            data = load_clean_data()
            st.session_state.clean_data = data
            return data
        except:
            # Fallback - ładuj bezpośrednio
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'flights_cleaned.csv')
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                data['flight_date'] = pd.to_datetime(data['flight_date'])
                st.session_state.clean_data = data
                return data
            else:
                st.error("❌ Nie można załadować danych")
                return None
    except Exception as e:
        st.error(f"❌ Błąd ładowania danych: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Ładuje modele ML"""
    try:
        classifier_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'notebooks', 'best_model_classifier.joblib')
        regressor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'notebooks', 'best_model_regressor.joblib')
        
        classifier = joblib.load(classifier_path) if os.path.exists(classifier_path) else None
        regressor = joblib.load(regressor_path) if os.path.exists(regressor_path) else None
        
        return classifier, regressor
    except Exception as e:
        st.error(f"❌ Błąd ładowania modeli: {str(e)}")
        return None, None

def calculate_kpis(data: pd.DataFrame) -> Dict[str, Any]:
    """Oblicza kluczowe wskaźniki KPI"""
    if data is None or data.empty:
        return {}
    
    # Punktualność (≤15 min opóźnienia)
    on_time_flights = (data['delay_minutes'] <= 15).sum()
    total_flights = len(data)
    on_time_percent = (on_time_flights / total_flights) * 100
    
    # Średnie opóźnienie
    avg_delay = data['delay_minutes'].mean()
    
    # Najgorsza linia
    airline_delays = data.groupby('airline')['delay_minutes'].agg(['mean', 'count']).reset_index()
    airline_delays.columns = ['airline', 'mean', 'count']
    airline_delays = airline_delays[airline_delays['count'] >= 50]  # Min 50 lotów
    if not airline_delays.empty:
        worst_airline_idx = airline_delays['mean'].idxmax()
        worst_airline = airline_delays.loc[worst_airline_idx, 'airline']
        worst_delay = airline_delays.loc[worst_airline_idx, 'mean']
    else:
        worst_airline = 'N/A'
        worst_delay = 0
    
    # Najgorsze lotnisko
    airport_delays = data.groupby('origin')['delay_minutes'].agg(['mean', 'count']).reset_index()
    airport_delays.columns = ['origin', 'mean', 'count']
    airport_delays = airport_delays[airport_delays['count'] >= 30]
    if not airport_delays.empty:
        worst_airport_idx = airport_delays['mean'].idxmax()
        worst_airport = airport_delays.loc[worst_airport_idx, 'origin']
    else:
        worst_airport = 'N/A'
    
    # Trend tygodniowy
    daily_delays = data.groupby(data['flight_date'].dt.date)['delay_minutes'].mean()
    if len(daily_delays) >= 14:
        trend = "📈 Rosnący" if daily_delays.tail(7).mean() > daily_delays.head(7).mean() else "📉 Malejący"
    else:
        trend = "📊 Brak danych"
    
    return {
        'on_time_percent': on_time_percent,
        'avg_delay': avg_delay,
        'worst_airline': worst_airline,
        'worst_delay': worst_delay,
        'worst_airport': worst_airport,
        'trend': trend,
        'total_flights': total_flights
    }

def create_timeline_chart(data: pd.DataFrame) -> go.Figure:
    """Tworzy interaktywny timeline opóźnień"""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    daily_stats = data.groupby(data['flight_date'].dt.date).agg({
        'delay_minutes': ['mean', 'count'],
        'flight_date': 'first'
    }).reset_index()
    
    daily_stats.columns = ['date', 'avg_delay', 'flight_count', 'flight_date_sample']
    daily_stats['on_time_percent'] = data.groupby(data['flight_date'].dt.date).apply(
        lambda x: (x['delay_minutes'] <= 15).sum() / len(x) * 100
    ).values
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Średnie Opóźnienie (minuty)', 'Punktualność (%)'),
        vertical_spacing=0.1
    )
    
    # Wykres opóźnień
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['avg_delay'],
            mode='lines+markers',
            name='Średnie opóźnienie',
            line=dict(color='#e74c3c', width=2),
            hovertemplate='Data: %{x}<br>Opóźnienie: %{y:.1f} min<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Wykres punktualności
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['on_time_percent'],
            mode='lines+markers',
            name='Punktualność',
            line=dict(color='#27ae60', width=2),
            hovertemplate='Data: %{x}<br>Punktualność: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        title_text="📈 Timeline Opóźnień - Analiza Czasowa",
        showlegend=False
    )
    
    return fig

# ===========================
# SIDEBAR NAWIGACJA
# ===========================
st.sidebar.title("🛩️ Nawigacja")
page = st.sidebar.selectbox(
    "Wybierz stronę:",
    ["🏠 Overview", "📊 Analytics", "🔮 Predictor"]
)

# Status połączenia z danymi
data = load_data()
classifier, regressor = load_models()

if data is not None:
    st.sidebar.success(f"✅ Dane załadowane: {len(data):,} lotów")
else:
    st.sidebar.error("❌ Problemy z danymi")

if classifier is not None and regressor is not None:
    st.sidebar.success("✅ Modele ML gotowe")
else:
    st.sidebar.warning("⚠️ Modele ML niedostępne")

# Model health check
try:
    health_status = model_health_check()
    status = health_status.get('overall_status', 'unknown')
    
    if status == 'healthy':
        st.sidebar.success("💚 System zdrowy")
    elif status == 'warning':
        st.sidebar.warning("💛 System wymaga uwagi")
        # Pokaż szczegóły warningów
        if 'checks' in health_status:
            warning_details = []
            for check_name, check_data in health_status['checks'].items():
                if check_data.get('status') == 'warning':
                    warning_details.append(f"• {check_data.get('message', check_name)}")
            if warning_details:
                with st.sidebar.expander("📋 Szczegóły ostrzeżeń"):
                    for detail in warning_details:
                        st.write(detail)
    elif status == 'unhealthy':
        st.sidebar.error("🔴 System niesprawny")
    else:
        st.sidebar.info(f"ℹ️ Status: {status}")
except Exception as e:
    st.sidebar.info(f"ℹ️ Błąd sprawdzania: {str(e)}")

# ===========================
# STRONA 1: OVERVIEW
# ===========================
if page == "🏠 Overview":
    st.markdown('<h1 class="main-header">✈️ Airline Analytics ML Dashboard</h1>', unsafe_allow_html=True)
    
    if data is None:
        st.error("❌ Brak danych do wyświetlenia")
        st.stop()
    
    # Oblicz KPI
    kpis = calculate_kpis(data)
    
    # ROW 1: KPI Cards
    st.subheader("📈 Kluczowe Wskaźniki Wydajności (KPI)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="metric-value success-metric">{kpis.get('on_time_percent', 0):.1f}%</div>
                <div class="metric-label">Loty punktualne (≤15 min)</div>
            </div>
            """, unsafe_allow_html=True
        )
    
    with col2:
        delay_class = "danger-metric" if kpis.get('avg_delay', 0) > 30 else "warning-metric" if kpis.get('avg_delay', 0) > 15 else "success-metric"
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="metric-value {delay_class}">{kpis.get('avg_delay', 0):.1f} min</div>
                <div class="metric-label">Średnie opóźnienie</div>
            </div>
            """, unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="metric-value danger-metric">{kpis.get('worst_airline', 'N/A')}</div>
                <div class="metric-label">Najgorsza linia ({kpis.get('worst_delay', 0):.1f} min)</div>
            </div>
            """, unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="metric-value">{kpis.get('trend', 'N/A')}</div>
                <div class="metric-label">Trend tygodniowy</div>
            </div>
            """, unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # ROW 2: Interactive Timeline
    st.subheader("📅 Interaktywny Timeline Opóźnień")
    
    # Selektor zakresu dat
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        start_date = st.date_input(
            "Data początkowa:",
            value=data['flight_date'].min().date(),
            min_value=data['flight_date'].min().date(),
            max_value=data['flight_date'].max().date()
        )
    
    with col2:
        end_date = st.date_input(
            "Data końcowa:",
            value=data['flight_date'].max().date(),
            min_value=data['flight_date'].min().date(),
            max_value=data['flight_date'].max().date()
        )
    
    # Filtruj dane według zakresu dat
    filtered_data = data[
        (data['flight_date'].dt.date >= start_date) & 
        (data['flight_date'].dt.date <= end_date)
    ]
    
    if not filtered_data.empty:
        timeline_fig = create_timeline_chart(filtered_data)
        st.plotly_chart(timeline_fig, use_container_width=True)
    else:
        st.warning("⚠️ Brak danych dla wybranego zakresu")
    
    st.markdown("---")
    
    # ROW 3: Real-time Prediction Simulator
    st.subheader("🎯 Symulator Przewidywań Real-time")
    
    if classifier is not None and regressor is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**🎲 Losowy lot z bazy danych:**")
            if st.button("🎰 Generuj losowy lot", type="primary"):
                # Wybierz losowy lot
                random_flight = data.sample(n=1).iloc[0]
                
                flight_details = {
                    'airline': random_flight['airline'],
                    'origin': random_flight['origin'],
                    'destination': random_flight['destination'],
                    'aircraft_type': random_flight.get('aircraft_type', 'B737'),
                    'distance_miles': random_flight.get('distance_miles', 1000),
                    'scheduled_departure': random_flight['scheduled_departure'],
                    'flight_date': random_flight['flight_date'].strftime('%Y-%m-%d')
                }
                
                st.session_state.random_flight = flight_details
        
        with col2:
            if hasattr(st.session_state, 'random_flight'):
                flight = st.session_state.random_flight
                
                st.write("**✈️ Szczegóły lotu:**")
                st.write(f"🏢 Linia: **{flight['airline']}**")
                st.write(f"🛫 Trasa: **{flight['origin']} → {flight['destination']}**")
                st.write(f"✈️ Samolot: **{flight['aircraft_type']}**")
                st.write(f"📅 Data: **{flight['flight_date']}**")
                st.write(f"⏰ Odlot: **{flight['scheduled_departure']}**")
                
                # Przewidywanie
                try:
                    prediction = predict_delay(flight, classifier, regressor)
                    if prediction:
                        is_delayed = prediction['is_delayed']
                        delay_minutes = prediction['expected_delay_minutes']
                        probability = prediction['delay_probability']
                        
                        if is_delayed:
                            st.error(f"🚫 **Przewidywane opóźnienie:** {delay_minutes:.0f} minut (prawdopodobieństwo: {probability:.1%})")
                        else:
                            st.success(f"✅ **Lot punktualny** (prawdopodobieństwo opóźnienia: {probability:.1%})")
                except Exception as e:
                    st.error(f"❌ Błąd przewidywania: {str(e)}")
    else:
        st.warning("⚠️ Modele ML niedostępne - nie można uruchomić symulatora")

# ===========================
# STRONA 2: ANALYTICS  
# ===========================
elif page == "📊 Analytics":
    st.markdown('<h1 class="main-header">📊 Zaawansowana Analityka</h1>', unsafe_allow_html=True)
    
    if data is None:
        st.error("❌ Brak danych do analizy")
        st.stop()
    
    # SIDEBAR FILTERS
    st.sidebar.subheader("🔍 Filtry")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Zakres dat:",
        value=[data['flight_date'].min().date(), data['flight_date'].max().date()],
        min_value=data['flight_date'].min().date(),
        max_value=data['flight_date'].max().date()
    )
    
    # Airline filter
    airlines = ['Wszystkie'] + sorted(data['airline'].unique().tolist())
    selected_airlines = st.sidebar.multiselect(
        "Linie lotnicze:",
        airlines,
        default=['Wszystkie']
    )
    
    # Airport filter
    airports = ['Wszystkie'] + sorted(data['origin'].unique().tolist())
    selected_airports = st.sidebar.multiselect(
        "Lotniska wylotu:",
        airports,
        default=['Wszystkie']
    )
    
    # Apply filters
    filtered_data = data.copy()
    
    # Upewnij się, że to DataFrame
    if not isinstance(filtered_data, pd.DataFrame):
        st.error("❌ Błąd: dane nie są w formacie DataFrame")
        st.stop()
    
    if len(date_range) == 2:
        filtered_data = filtered_data[
            (filtered_data['flight_date'].dt.date >= date_range[0]) &
            (filtered_data['flight_date'].dt.date <= date_range[1])
        ]
    
    if 'Wszystkie' not in selected_airlines and selected_airlines:
        filtered_data = filtered_data[filtered_data['airline'].isin(selected_airlines)]
    
    if 'Wszystkie' not in selected_airports and selected_airports:
        filtered_data = filtered_data[filtered_data['origin'].isin(selected_airports)]
    
    if filtered_data.empty:
        st.warning("⚠️ Brak danych dla wybranych filtrów")
        st.stop()
    
    st.info(f"📊 Analizuję {len(filtered_data):,} lotów po zastosowaniu filtrów")
    
    # ROW 1: Dynamic Charts
    st.subheader("📈 Wykresy Dynamiczne")
    
    tab1, tab2, tab3 = st.tabs(["Analiza linii lotniczych", "Analiza lotnisk", "Analiza czasowa"])
    
    with tab1:
        try:
            # Analiza linii lotniczych
            airline_stats = filtered_data.groupby('airline').agg({
                'delay_minutes': ['mean', 'count', 'std'],
                'flight_date': 'first'
            }).reset_index()
            airline_stats.columns = ['airline', 'avg_delay', 'flight_count', 'delay_std', 'first_flight']
            
            # Bezpieczne obliczenie punktualności
            on_time_by_airline = filtered_data.groupby('airline').apply(
                lambda x: (x['delay_minutes'] <= 15).sum() / len(x) * 100
            ).reset_index()
            on_time_by_airline.columns = ['airline', 'on_time_percent']
            
            airline_stats = airline_stats.merge(on_time_by_airline, on='airline')
            
            # Filtruj linie z min. 20 lotami
            airline_stats = airline_stats[airline_stats['flight_count'] >= 20].sort_values('avg_delay', ascending=False)
            
            if airline_stats.empty:
                st.warning("⚠️ Brak linii lotniczych z minimum 20 lotami")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart - średnie opóźnienia
                    fig1 = px.bar(
                        airline_stats.head(10),
                        x='avg_delay',
                        y='airline',
                        orientation='h',
                        title='Top 10 Linii - Średnie Opóźnienie',
                        labels={'avg_delay': 'Średnie opóźnienie (min)', 'airline': 'Linia lotnicza'},
                        color='avg_delay',
                        color_continuous_scale='Reds'
                    )
                    fig1.update_layout(height=400)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Scatter plot - punktualność vs liczba lotów
                    fig2 = px.scatter(
                        airline_stats,
                        x='flight_count',
                        y='on_time_percent',
                        size='delay_std',
                        hover_name='airline',
                        title='Punktualność vs Liczba Lotów',
                        labels={
                            'flight_count': 'Liczba lotów',
                            'on_time_percent': 'Punktualność (%)',
                            'delay_std': 'Odchylenie std opóźnień'
                        }
                    )
                    fig2.update_layout(height=400)
                    st.plotly_chart(fig2, use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Błąd analizy linii lotniczych: {str(e)}")
            st.info("📊 Wyświetlam podstawowe statystyki...")
            
            # Prosty wykres jako fallback
            if isinstance(filtered_data, pd.DataFrame):
                simple_stats = filtered_data.groupby('airline')['delay_minutes'].mean().reset_index()
                fig_simple = px.bar(simple_stats, x='airline', y='delay_minutes', 
                                  title='Średnie Opóźnienia - Linie Lotnicze')
                st.plotly_chart(fig_simple, use_container_width=True)
            else:
                st.error("❌ Nie można wyświetlić wykres - nieprawidłowy format danych")
    
    with tab2:
        # Analiza lotnisk
        if isinstance(filtered_data, pd.DataFrame):
            airport_stats = filtered_data.groupby('origin').agg({
                'delay_minutes': ['mean', 'count'],
                'flight_date': 'first'
            }).reset_index()
            airport_stats.columns = ['airport', 'avg_delay', 'flight_count', 'first_flight']
            airport_stats = airport_stats[airport_stats['flight_count'] >= 15].sort_values('avg_delay', ascending=False)
            
            # Mapa lotnisk (symulacja - w rzeczywistości potrzebne byłyby współrzędne)
            fig3 = px.bar(
                airport_stats.head(15),
                x='airport',
                y='avg_delay',
                title='Top 15 Lotnisk - Średnie Opóźnienie',
                labels={'avg_delay': 'Średnie opóźnienie (min)', 'airport': 'Lotnisko'},
                color='avg_delay',
                color_continuous_scale='Blues'
            )
            fig3.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Tabela szczegółów
            st.subheader("📋 Szczegóły lotnisk")
            st.dataframe(
                airport_stats.head(10)[['airport', 'avg_delay', 'flight_count']].round(2),
                use_container_width=True
            )
        else:
            st.error("❌ Nie można wyświetlić analizy lotnisk - nieprawidłowy format danych")
    
    with tab3:
        # Analiza czasowa
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                # Heatmapa - dzień tygodnia vs godzina
                if isinstance(filtered_data, pd.DataFrame):
                    time_data = filtered_data.copy()
                    
                    # Bezpieczne wyciąganie godziny z scheduled_departure
                    if 'hour' in time_data.columns:
                        time_data['hour_analysis'] = time_data['hour']
                    else:
                        # Jeśli scheduled_departure to string w formacie HH:MM
                        time_data['hour_analysis'] = time_data['scheduled_departure'].astype(str).str.split(':').str[0].astype(int)
                    
                    time_data['day_name'] = time_data['flight_date'].dt.day_name()
                    
                    heatmap_data = time_data.groupby(['day_name', 'hour_analysis'])['delay_minutes'].mean().reset_index()
                
                    if not heatmap_data.empty:
                        heatmap_pivot = heatmap_data.pivot(index='day_name', columns='hour_analysis', values='delay_minutes')
                        
                        fig4 = px.imshow(
                            heatmap_pivot,
                            title='Heatmapa Opóźnień - Dzień vs Godzina',
                            labels={'color': 'Średnie opóźnienie (min)'},
                            aspect='auto'
                        )
                        fig4.update_layout(height=400)
                        st.plotly_chart(fig4, use_container_width=True)
                    else:
                        st.warning("⚠️ Brak danych do utworzenia heatmapy")
                else:
                    st.error("❌ Nie można utworzyć heatmapy - nieprawidłowy format danych")
                    
            except Exception as e:
                st.error(f"❌ Błąd tworzenia heatmapy: {str(e)}")
                st.info("📊 Wyświetlam alternatywny wykres...")
                
                # Prosty wykres jako fallback
                if isinstance(filtered_data, pd.DataFrame) and 'hour' in filtered_data.columns:
                    hourly_delays = filtered_data.groupby('hour')['delay_minutes'].mean().reset_index()
                    fig4_alt = px.bar(hourly_delays, x='hour', y='delay_minutes', 
                                    title='Średnie Opóźnienia według Godziny')
                    st.plotly_chart(fig4_alt, use_container_width=True)
                else:
                    st.error("❌ Nie można wyświetlić alternatywnego wykresu")
        
        with col2:
            try:
                # Box plot - opóźnienia według dnia tygodnia
                if isinstance(filtered_data, pd.DataFrame):
                    time_data = filtered_data.copy()
                    time_data['day_name'] = time_data['flight_date'].dt.day_name()
                    
                    fig5 = px.box(
                        time_data,
                        x='day_name',
                        y='delay_minutes',
                        title='Rozkład Opóźnień według Dnia Tygodnia'
                    )
                    fig5.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig5, use_container_width=True)
                else:
                    st.error("❌ Nie można utworzyć box plot - nieprawidłowy format danych")
                
            except Exception as e:
                st.error(f"❌ Błąd tworzenia box plot: {str(e)}")
                
                # Prosty wykres jako fallback
                if isinstance(filtered_data, pd.DataFrame):
                    daily_delays = filtered_data.groupby(filtered_data['flight_date'].dt.day_name())['delay_minutes'].mean().reset_index()
                    daily_delays.columns = ['day_name', 'avg_delay']
                    fig5_alt = px.bar(daily_delays, x='day_name', y='avg_delay', 
                                    title='Średnie Opóźnienia według Dnia')
                    st.plotly_chart(fig5_alt, use_container_width=True)
                else:
                    st.error("❌ Nie można wyświetlić alternatywnego wykresu")
    
    st.markdown("---")
    
    # ROW 2: Comparison Tool
    st.subheader("⚖️ Narzędzie Porównawcze")
    
    comparison_type = st.radio("Typ porównania:", ["Linie lotnicze", "Lotniska"], horizontal=True)
    
    col1, col2 = st.columns(2)
    
    if comparison_type == "Linie lotnicze":
        if isinstance(filtered_data, pd.DataFrame):
            with col1:
                airline1 = st.selectbox("Wybierz pierwszą linię:", sorted(filtered_data['airline'].unique()))
            with col2:
                airline2 = st.selectbox("Wybierz drugą linię:", sorted(filtered_data['airline'].unique()))
        else:
            st.error("❌ Nie można wyświetlić porównania - nieprawidłowy format danych")
            airline1 = airline2 = None
        
        if airline1 != airline2:
            # Porównaj linie
            data1 = filtered_data[filtered_data['airline'] == airline1]
            data2 = filtered_data[filtered_data['airline'] == airline2]
            
            comparison_fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f'{airline1}', f'{airline2}'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )

            # Histogramy opóźnień
            comparison_fig.add_trace(
                go.Histogram(x=data1['delay_minutes'], name=airline1, opacity=0.7, nbinsx=30),
                row=1, col=1
            )
            comparison_fig.add_trace(
                go.Histogram(x=data2['delay_minutes'], name=airline2, opacity=0.7, nbinsx=30),
                row=1, col=2
            )
            
            comparison_fig.update_layout(
                title_text=f"📊 Porównanie Rozkładu Opóźnień: {airline1} vs {airline2}",
                height=400
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Statystyki porównawcze
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    f"Średnie opóźnienie - {airline1}",
                    f"{data1['delay_minutes'].mean():.1f} min",
                    f"{data1['delay_minutes'].mean() - data2['delay_minutes'].mean():.1f} min"
                )
            with col2:
                st.metric(
                    f"Punktualność - {airline1}",
                    f"{(data1['delay_minutes'] <= 15).mean() * 100:.1f}%",
                    f"{((data1['delay_minutes'] <= 15).mean() - (data2['delay_minutes'] <= 15).mean()) * 100:.1f}%"
                )
            with col3:
                st.metric(f"Liczba lotów - {airline1}", f"{len(data1):,}")
    
    else:  # Lotniska
        if isinstance(filtered_data, pd.DataFrame):
            with col1:
                airport1 = st.selectbox("Wybierz pierwsze lotnisko:", sorted(filtered_data['origin'].unique()))
            with col2:
                airport2 = st.selectbox("Wybierz drugie lotnisko:", sorted(filtered_data['origin'].unique()))
        else:
            st.error("❌ Nie można wyświetlić porównania - nieprawidłowy format danych")
            airport1 = airport2 = None
        
        if airport1 != airport2:
            # Analogiczne porównanie dla lotnisk
            data1 = filtered_data[filtered_data['origin'] == airport1]
            data2 = filtered_data[filtered_data['origin'] == airport2]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    f"Średnie opóźnienie - {airport1}",
                    f"{data1['delay_minutes'].mean():.1f} min",
                    f"{data1['delay_minutes'].mean() - data2['delay_minutes'].mean():.1f} min"
                )
            with col2:
                st.metric(
                    f"Punktualność - {airport1}",
                    f"{(data1['delay_minutes'] <= 15).mean() * 100:.1f}%",
                    f"{((data1['delay_minutes'] <= 15).mean() - (data2['delay_minutes'] <= 15).mean()) * 100:.1f}%"
                )
            with col3:
                st.metric(f"Liczba lotów - {airport1}", f"{len(data1):,}")

# ===========================
# STRONA 3: PREDICTOR
# ===========================
elif page == "🔮 Predictor":
    st.markdown('<h1 class="main-header">🔮 Przewidywanie Opóźnień</h1>', unsafe_allow_html=True)
    
    if classifier is None or regressor is None:
        st.error("❌ Modele ML niedostępne. Sprawdź czy pliki modeli istnieją.")
        st.stop()
    
    # ROW 1: Prediction Form
    st.subheader("✈️ Wprowadź dane lotu")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        airline = st.selectbox(
            "Linia lotnicza:",
            sorted(data['airline'].unique()) if data is not None else ['American Airlines', 'Delta', 'United']
        )
        
        origin = st.selectbox(
            "Lotnisko wylotu:",
            sorted(data['origin'].unique()) if data is not None else ['JFK', 'LAX', 'ORD']
        )
        
        flight_date = st.date_input(
            "Data lotu:",
            value=datetime.now().date(),
            min_value=datetime.now().date()
        )
    
    with col2:
        destination = st.selectbox(
            "Lotnisko docelowe:",
            sorted(data['destination'].unique()) if data is not None else ['JFK', 'LAX', 'ORD']
        )
        
        aircraft_type = st.selectbox(
            "Typ samolotu:",
            ['B737', 'A320', 'B777', 'A330', 'B747', 'A350', 'B787']
        )
        
        scheduled_departure = st.time_input(
            "Planowany odlot:",
            value=datetime.now().time()
        )
    
    with col3:
        distance_miles = st.number_input(
            "Odległość (mile):",
            min_value=100,
            max_value=5000,
            value=1000,
            step=50
        )
        
        # Dodatkowe opcje
        weather_conditions = st.selectbox(
            "Warunki pogodowe:",
            ['Clear', 'Cloudy', 'Rainy', 'Stormy', 'Foggy']
        )
    
    # Predict Button
    if st.button("🔮 Przewiduj opóźnienie", type="primary", use_container_width=True):
        try:
            # Przygotuj dane lotu
            flight_details = {
                'airline': airline,
                'origin': origin,
                'destination': destination,
                'aircraft_type': aircraft_type,
                'distance_miles': distance_miles,
                'scheduled_departure': scheduled_departure.strftime('%H:%M'),
                'flight_date': flight_date.strftime('%Y-%m-%d')
            }
            
            # Przewidywanie
            prediction = predict_delay(flight_details, classifier, regressor)
            
            if prediction:
                st.markdown("---")
                st.subheader("🎯 Wynik Przewidywania")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    is_delayed = prediction['is_delayed']
                    if is_delayed:
                        st.error("🚫 **OPÓŹNIENIE PRZEWIDYWANE**")
                    else:
                        st.success("✅ **LOT PUNKTUALNY**")
                
                with col2:
                    delay_minutes = prediction['expected_delay_minutes']
                    st.metric(
                        "Przewidywane opóźnienie:",
                        f"{delay_minutes:.0f} minut"
                    )
                
                with col3:
                    probability = prediction['delay_probability']
                    st.metric(
                        "Prawdopodobieństwo opóźnienia:",
                        f"{probability:.1%}"
                    )
                
                # Confidence indicator
                confidence_level = "Wysokie" if probability > 0.8 or probability < 0.2 else "Średnie" if probability > 0.6 or probability < 0.4 else "Niskie"
                confidence_color = "success" if confidence_level == "Wysokie" else "warning" if confidence_level == "Średnie" else "error"
                
                st.info(f"🎯 **Pewność przewidywania:** {confidence_level}")
                
        except Exception as e:
            st.error(f"❌ Błąd podczas przewidywania: {str(e)}")
    
    st.markdown("---")
    
    # ROW 2: Similar Flights Analysis
    st.subheader("🔍 Analiza Podobnych Lotów")
    
    if data is not None and st.button("🔍 Znajdź podobne loty"):
        try:
            # Filtruj podobne loty (ta sama trasa, linia, typ samolotu)
            similar_flights = data[
                (data['airline'] == airline) &
                (data['origin'] == origin) &
                (data['destination'] == destination)
            ]
            
            if len(similar_flights) > 0:
                st.success(f"✅ Znaleziono {len(similar_flights)} podobnych lotów")
                
                # Statystyki podobnych lotów
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_delay = similar_flights['delay_minutes'].mean()
                    st.metric("Średnie opóźnienie", f"{avg_delay:.1f} min")
                
                with col2:
                    on_time_rate = (similar_flights['delay_minutes'] <= 15).mean() * 100
                    st.metric("Punktualność", f"{on_time_rate:.1f}%")
                
                with col3:
                    max_delay = similar_flights['delay_minutes'].max()
                    st.metric("Max opóźnienie", f"{max_delay:.0f} min")
                
                with col4:
                    if isinstance(similar_flights, pd.DataFrame):
                        median_delay = similar_flights['delay_minutes'].median()
                        st.metric("Mediana opóźnienia", f"{median_delay:.1f} min")
                    else:
                        st.metric("Mediana opóźnienia", "N/A")
                
                # Histogram opóźnień podobnych lotów
                fig_similar = px.histogram(
                    similar_flights,
                    x='delay_minutes',
                    nbins=30,
                    title=f'Rozkład Opóźnień - Podobne Loty ({airline}: {origin}→{destination})',
                    labels={'delay_minutes': 'Opóźnienie (minuty)', 'count': 'Liczba lotów'}
                )
                fig_similar.add_vline(
                    x=avg_delay,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Średnia: {avg_delay:.1f} min"
                )
                st.plotly_chart(fig_similar, use_container_width=True)
                
                # Tabela ostatnich podobnych lotów
                st.subheader("📋 Ostatnie podobne loty")
                if isinstance(similar_flights, pd.DataFrame):
                    recent_similar = similar_flights.sort_values('flight_date', ascending=False).head(10)
                    display_cols = ['flight_date', 'scheduled_departure', 'delay_minutes', 'delay_reason']
                    available_cols = [col for col in display_cols if col in recent_similar.columns]
                    st.dataframe(recent_similar[available_cols], use_container_width=True)
                else:
                    st.error("❌ Nie można wyświetlić tabeli podobnych lotów")
                
            else:
                st.warning("⚠️ Nie znaleziono podobnych lotów w bazie danych")
                
        except Exception as e:
            st.error(f"❌ Błąd podczas wyszukiwania podobnych lotów: {str(e)}")

# ===========================
# FOOTER
# ===========================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        ✈️ <strong>Airline Analytics ML Dashboard</strong> | 
        Powered by Streamlit & Python | 
        © 2025 UFEQ
    </div>
    """, 
    unsafe_allow_html=True
) 