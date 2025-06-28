"""
🇪🇺 EUROPEAN AIRLINE ANALYTICS - POPRAWIONY DASHBOARD 
====================================================

Interaktywny dashboard Streamlit dla poprawionych modeli europejskich:
✅ Bez data leakage
✅ Class imbalance naprawiony  
✅ Dane europejskie/polskie
✅ Realistyczne przewidywania

Autorzy: AirlineAnalytics-ML Team
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
    from european_models import EuropeanDelayClassifier, EuropeanDelayRegressor, predict_european_delay
    from european_analysis import analyze_polish_routes, analyze_european_carriers
except ImportError as e:
    st.error(f"❌ Błąd importu modułów europejskich: {str(e)}")
    st.error("Upewnij się, że wszystkie pliki europejskie są dostępne.")
    st.stop()

# ===========================
# KONFIGURACJA STRONY
# ===========================
st.set_page_config(
    page_title="🇪🇺 European Airline Analytics - Poprawiony Dashboard",
    page_icon="🇪🇺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS dla europejskiego stylu
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #003d82;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #003d82, #ffcc00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .eu-kpi-container {
        background: linear-gradient(135deg, #f0f8ff, #e6f3ff);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 8px solid #003d82;
        margin: 0.8rem 0;
        box-shadow: 0 4px 6px rgba(0, 61, 130, 0.1);
    }
    .polish-accent {
        color: #dc143c;
        font-weight: bold;
    }
    .eu-metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #003d82;
    }
    .fixed-badge {
        background: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
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
# FUNKCJE ŁADOWANIA DANYCH
# ===========================

@st.cache_data(ttl=3600)
def load_european_data():
    """Ładuje dane europejskie"""
    try:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'european_flights_data.csv')
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            data['flight_date'] = pd.to_datetime(data['flight_date'])
            return data
        else:
            st.error("❌ Brak danych europejskich! Uruchom: python demo_european_analysis.py")
            return None
    except Exception as e:
        st.error(f"❌ Błąd ładowania danych europejskich: {str(e)}")
        return None

@st.cache_resource
def load_fixed_european_models():
    """Ładuje poprawione modele europejskie"""
    try:
        base_path = os.path.dirname(os.path.dirname(__file__))
        classifier_path = os.path.join(base_path, 'notebooks', 'european_fixed_model_classifier.joblib')
        regressor_path = os.path.join(base_path, 'notebooks', 'european_fixed_model_regressor.joblib')
        
        classifier = None
        regressor = None
        
        if os.path.exists(classifier_path):
            classifier = joblib.load(classifier_path)
            st.success("✅ Poprawiony klasyfikator europejski załadowany")
        else:
            st.warning("⚠️ Brak poprawionego klasyfikatora! Uruchom: python train_european_models_fixed.py")
        
        if os.path.exists(regressor_path):
            regressor = joblib.load(regressor_path)  
            st.success("✅ Poprawiony regressor europejski załadowany")
        else:
            st.warning("⚠️ Brak poprawionego regressora!")
            
        return classifier, regressor
    except Exception as e:
        st.error(f"❌ Błąd ładowania poprawionych modeli: {str(e)}")
        return None, None

# ===========================
# FUNKCJE KPI EUROPEJSKICH
# ===========================

def calculate_european_kpis(data: pd.DataFrame) -> Dict[str, Any]:
    """Oblicza KPI dla danych europejskich"""
    if data is None or data.empty:
        return {}
    
    # Podstawowe statystyki
    total_flights = len(data)
    delayed_flights = len(data[data['delay_minutes'] > 15])
    on_time_percent = ((total_flights - delayed_flights) / total_flights) * 100
    avg_delay = data['delay_minutes'].mean()
    
    # Statystyki polskie
    polish_flights = len(data[(data['country_origin'] == 'Polska') | (data['country_destination'] == 'Polska')])
    polish_percent = (polish_flights / total_flights) * 100
    
    # LOT vs konkurencja
    lot_flights = data[data['airline'] == 'LOT Polish Airlines']
    lot_avg_delay = lot_flights['delay_minutes'].mean() if len(lot_flights) > 0 else 0
    
    # Najgorsza linia europejska
    airline_stats = data.groupby('airline').agg({
        'delay_minutes': 'mean',
        'airline': 'count'
    }).rename(columns={'airline': 'count'})
    airline_stats = airline_stats[airline_stats['count'] >= 100]
    
    if not airline_stats.empty:
        worst_airline = airline_stats['delay_minutes'].idxmax()
        worst_delay = airline_stats.loc[worst_airline, 'delay_minutes']
    else:
        worst_airline = 'N/A'
        worst_delay = 0
    
    # Problematyczne lotniska europejskie
    problematic_airports = ['LHR', 'CDG', 'FRA', 'AMS']
    problematic_flights = data[data['origin'].isin(problematic_airports) | 
                             data['destination'].isin(problematic_airports)]
    problematic_delay = problematic_flights['delay_minutes'].mean()
    
    return {
        'total_flights': total_flights,
        'on_time_percent': on_time_percent,
        'avg_delay': avg_delay,
        'polish_flights': polish_flights,
        'polish_percent': polish_percent,
        'lot_avg_delay': lot_avg_delay,
        'worst_airline': worst_airline,
        'worst_delay': worst_delay,
        'problematic_delay': problematic_delay,
        'delayed_flights': delayed_flights
    }

def create_european_overview_charts(data: pd.DataFrame):
    """Tworzy wykresy overview dla danych europejskich"""
    
    # 1. Timeline opóźnień europejskich
    daily_stats = data.groupby(data['flight_date'].dt.date).agg({
        'delay_minutes': ['mean', 'count']
    }).reset_index()
    daily_stats.columns = ['date', 'avg_delay', 'flight_count']
    
    fig_timeline = px.line(daily_stats, x='date', y='avg_delay', 
                          title='📈 Timeline Opóźnień Europejskich',
                          labels={'avg_delay': 'Średnie opóźnienie (min)', 'date': 'Data'})
    fig_timeline.update_layout(height=400)
    
    # 2. Porównanie krajów
    country_stats = data.groupby('country_origin').agg({
        'delay_minutes': ['mean', 'count']
    }).reset_index()
    country_stats.columns = ['country', 'avg_delay', 'count']
    country_stats = country_stats[country_stats['count'] >= 50].sort_values('avg_delay', ascending=True)
    
    fig_countries = px.bar(country_stats, x='avg_delay', y='country', 
                          title='🌍 Opóźnienia według Krajów Europejskich',
                          labels={'avg_delay': 'Średnie opóźnienie (min)', 'country': 'Kraj'},
                          color='avg_delay', color_continuous_scale='RdYlBu_r')
    fig_countries.update_layout(height=500)
    
    return fig_timeline, fig_countries

def create_polish_analysis_charts(data: pd.DataFrame):
    """Tworzy wykresy analizy polskiej"""
    
    # Loty polskie vs międzynarodowe
    data['flight_type'] = data.apply(
        lambda row: '🇵🇱 Polski' if row['country_origin'] == 'Polska' and row['country_destination'] == 'Polska'
        else '🇪🇺 Z Polski' if row['country_origin'] == 'Polska'
        else '🇪🇺 Do Polski' if row['country_destination'] == 'Polska'
        else '🌍 Międzynarodowy', axis=1
    )
    
    flight_type_stats = data.groupby('flight_type').agg({
        'delay_minutes': ['mean', 'count']
    }).reset_index()
    flight_type_stats.columns = ['flight_type', 'avg_delay', 'count']
    
    fig_polish = px.bar(flight_type_stats, x='flight_type', y='avg_delay',
                       title='🇵🇱 Analiza Lotów Polskich vs Europejskich',
                       labels={'avg_delay': 'Średnie opóźnienie (min)', 'flight_type': 'Typ lotu'},
                       color='avg_delay', color_continuous_scale='RdYlBu_r')
    
    # LOT vs konkurencja
    airlines_comparison = data.groupby('airline').agg({
        'delay_minutes': ['mean', 'count']
    }).reset_index()
    airlines_comparison.columns = ['airline', 'avg_delay', 'count']
    airlines_comparison = airlines_comparison[airlines_comparison['count'] >= 100]
    airlines_comparison['is_lot'] = airlines_comparison['airline'] == 'LOT Polish Airlines'
    
    fig_lot = px.scatter(airlines_comparison, x='count', y='avg_delay', 
                        color='is_lot', size='count',
                        title='✈️ LOT vs Konkurencja Europejska',
                        labels={'avg_delay': 'Średnie opóźnienie (min)', 'count': 'Liczba lotów'},
                        hover_data=['airline'])
    
    return fig_polish, fig_lot

# ===========================
# GŁÓWNA APLIKACJA
# ===========================

def main():
    # Header
    st.markdown('<h1 class="main-header">🇪🇺 European Airline Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<center><span class="fixed-badge">✅ POPRAWIONY - BEZ DATA LEAKAGE</span></center>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.markdown("## 🛠️ Konfiguracja")
    st.sidebar.markdown('<span class="warning-fixed">⚠️ Używa poprawionych modeli!</span>', unsafe_allow_html=True)
    
    # Wybór strony
    page = st.sidebar.selectbox(
        "📍 Wybierz stronę:",
        ["🏠 Overview", "📊 Analiza Europejska", "🔮 Przewidywanie (Poprawione)", "🧪 Testy Modeli"]
    )
    
    # Załaduj dane i modele
    data = load_european_data()
    classifier, regressor = load_fixed_european_models()
    
    if data is None:
        st.error("❌ BŁĄD: Brak danych europejskich!")
        st.info("🔧 Uruchom: `python demo_european_analysis.py`")
        return
    
    # Informacje o modelach
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Status Modeli")
    if classifier:
        st.sidebar.success(f"✅ Klasyfikator: {classifier.model_type}")
        st.sidebar.info(f"📊 Features: {len(classifier.feature_names) if hasattr(classifier, 'feature_names') else 'N/A'}")
    else:
        st.sidebar.error("❌ Brak klasyfikatora")
    
    if regressor:
        st.sidebar.success(f"✅ Regressor: {regressor.model_type}")
    else:
        st.sidebar.error("❌ Brak regressora")
    
    # Wyświetlanie strony
    if page == "🏠 Overview":
        show_overview_page(data)
    elif page == "📊 Analiza Europejska":
        show_european_analysis_page(data)
    elif page == "🔮 Przewidywanie (Poprawione)":
        show_prediction_page(data, classifier, regressor)
    elif page == "🧪 Testy Modeli":
        show_model_tests_page(data, classifier, regressor)

def show_overview_page(data: pd.DataFrame):
    """Strona przeglądu europejskiego"""
    st.markdown("## 🏠 Przegląd Europejski")
    
    # KPIs
    kpis = calculate_european_kpis(data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="eu-kpi-container">
            <div class="eu-metric-value">{kpis.get('total_flights', 0):,}</div>
            <div class="metric-label">🇪🇺 Łączne loty europejskie</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        on_time = kpis.get('on_time_percent', 0)
        color_class = "success-metric" if on_time >= 80 else "warning-metric" if on_time >= 70 else "danger-metric"
        st.markdown(f"""
        <div class="eu-kpi-container">
            <div class="eu-metric-value {color_class}">{on_time:.1f}%</div>
            <div class="metric-label">⏰ Punktualność europejska</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="eu-kpi-container">
            <div class="eu-metric-value polish-accent">{kpis.get('polish_percent', 0):.1f}%</div>
            <div class="metric-label">🇵🇱 Loty polskie</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        lot_delay = kpis.get('lot_avg_delay', 0)
        st.markdown(f"""
        <div class="eu-kpi-container">
            <div class="eu-metric-value">{lot_delay:.1f} min</div>
            <div class="metric-label">✈️ LOT średnie opóźnienie</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Wykresy
    col1, col2 = st.columns(2)
    
    with col1:
        fig_timeline, fig_countries = create_european_overview_charts(data)
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_countries, use_container_width=True)
    
    # Dodatkowe informacje
    st.markdown("### 📋 Podsumowanie Europejskie")
    st.info(f"""
    **🎯 Statystyki kluczowe:**
    - Najgorsza linia: **{kpis.get('worst_airline', 'N/A')}** ({kpis.get('worst_delay', 0):.1f} min)
    - Loty opóźnione: **{kpis.get('delayed_flights', 0):,}** z {kpis.get('total_flights', 0):,}
    - Problematyczne huby: **{kpis.get('problematic_delay', 0):.1f} min** średnio
    """)

def show_european_analysis_page(data: pd.DataFrame):
    """Strona analizy europejskiej"""
    st.markdown("## 📊 Analiza Europejska")
    
    # Filtry
    st.markdown("### 🔍 Filtry")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        countries = ['Wszystkie'] + sorted(data['country_origin'].unique().tolist())
        country_filter = st.selectbox("🌍 Kraj pochodzenia:", countries)
    
    with col2:
        airlines = ['Wszystkie'] + sorted(data['airline'].unique().tolist())
        airline_filter = st.selectbox("✈️ Linia lotnicza:", airlines)
    
    with col3:
        date_range = st.date_input("📅 Zakres dat:", 
                                  value=[data['flight_date'].min(), data['flight_date'].max()],
                                  min_value=data['flight_date'].min(),
                                  max_value=data['flight_date'].max())
    
    # Filtrowanie danych
    filtered_data = data.copy()
    if country_filter != 'Wszystkie':
        filtered_data = filtered_data[filtered_data['country_origin'] == country_filter]
    if airline_filter != 'Wszystkie':
        filtered_data = filtered_data[filtered_data['airline'] == airline_filter]
    if len(date_range) == 2:
        filtered_data = filtered_data[
            (filtered_data['flight_date'] >= pd.Timestamp(date_range[0])) &
            (filtered_data['flight_date'] <= pd.Timestamp(date_range[1]))
        ]
    
    st.markdown(f"📊 **Filtrowane dane**: {len(filtered_data):,} lotów")
    
    # Wykresy polskie
    st.markdown("---")
    fig_polish, fig_lot = create_polish_analysis_charts(filtered_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_polish, use_container_width=True)
    with col2:
        st.plotly_chart(fig_lot, use_container_width=True)

def show_prediction_page(data: pd.DataFrame, classifier, regressor):
    """Strona przewidywania z poprawionymi modelami"""
    st.markdown("## 🔮 Przewidywanie Opóźnień (Poprawione)")
    st.markdown('<span class="fixed-badge">✅ BEZ DATA LEAKAGE</span>', unsafe_allow_html=True)
    
    if not classifier or not regressor:
        st.error("❌ Brak poprawionych modeli! Uruchom: `python train_european_models_fixed.py`")
        return
    
    st.markdown("### ✈️ Parametry lotu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Podstawowe parametry
        flight_date = st.date_input("📅 Data lotu:", datetime.now().date())
        airline = st.selectbox("✈️ Linia lotnicza:", sorted(data['airline'].unique()))
        origin = st.selectbox("🛫 Origin:", sorted(data['origin'].unique()))
        destination = st.selectbox("🛬 Destination:", sorted(data['destination'].unique()))
    
    with col2:
        # Szczegóły czasowe
        departure_time = st.time_input("⏰ Godzina odlotu:", datetime.now().time())
        
        # Automatyczne wypełnianie na podstawie danych
        route_data = data[(data['origin'] == origin) & (data['destination'] == destination)]
        if not route_data.empty:
            avg_distance = route_data['distance_km'].mean()
            common_country_origin = route_data['country_origin'].mode()[0] if not route_data['country_origin'].mode().empty else 'Polska'
            common_country_dest = route_data['country_destination'].mode()[0] if not route_data['country_destination'].mode().empty else 'Niemcy'
        else:
            avg_distance = 1000
            common_country_origin = 'Polska'
            common_country_dest = 'Niemcy'
        
        distance_km = st.number_input("📏 Dystans (km):", value=int(avg_distance), min_value=100, max_value=3000)
        country_origin = st.selectbox("🌍 Kraj origin:", sorted(data['country_origin'].unique()), 
                                     index=sorted(data['country_origin'].unique()).index(common_country_origin))
        country_destination = st.selectbox("🌍 Kraj destination:", sorted(data['country_destination'].unique()),
                                          index=sorted(data['country_destination'].unique()).index(common_country_dest))
    
    # Przycisk przewidywania
    if st.button("🔮 Przewiduj opóźnienie", type="primary"):
        # Przygotuj dane
        flight_details = {
            'flight_date': flight_date.strftime('%Y-%m-%d'),
            'airline': airline,
            'origin': origin,
            'destination': destination,
            'country_origin': country_origin,
            'country_destination': country_destination,
            'distance_km': distance_km,
            'scheduled_departure': departure_time.strftime('%H:%M'),
            'day_of_week': flight_date.weekday(),
            'month': flight_date.month,
            'hour': departure_time.hour
        }
        
        try:
            # Przewidywanie
            with st.spinner('🔄 Przewidywanie...'):
                prediction = predict_european_delay(flight_details, classifier, regressor)
            
            # Wyniki
            st.markdown("---")
            st.markdown("### 📊 Wyniki Przewidywania")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                prob = prediction['delay_probability'] * 100
                color = "danger-metric" if prob > 70 else "warning-metric" if prob > 30 else "success-metric"
                st.markdown(f"""
                <div class="eu-kpi-container">
                    <div class="eu-metric-value {color}">{prob:.1f}%</div>
                    <div class="metric-label">🎯 Prawdopodobieństwo opóźnienia</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                status = "OPÓŹNIONY" if prediction['is_delayed'] else "PUNKTUALNY"
                status_color = "danger-metric" if prediction['is_delayed'] else "success-metric"
                st.markdown(f"""
                <div class="eu-kpi-container">
                    <div class="eu-metric-value {status_color}">{status}</div>
                    <div class="metric-label">🚨 Przewidywany status</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                delay_min = prediction['predicted_delay_minutes']
                st.markdown(f"""
                <div class="eu-kpi-container">
                    <div class="eu-metric-value">{delay_min:.0f} min</div>
                    <div class="metric-label">⏰ Przewidywane opóźnienie</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Dodatkowe info
            risk_level = prediction['delay_risk']
            risk_color = {"Wysokie": "🔴", "Średnie": "🟡", "Niskie": "🟢"}
            st.info(f"""
            **📋 Szczegóły przewidywania:**
            - Ryzyko: {risk_color.get(risk_level, '⚪')} **{risk_level}**
            - Kategoria: **{prediction.get('delay_category', 'N/A')}**
            - Trasa: **{origin} → {destination}** ({distance_km} km)
            - Model: **Poprawiony europejski** ✅
            """)
            
        except Exception as e:
            st.error(f"❌ Błąd przewidywania: {str(e)}")

def show_model_tests_page(data: pd.DataFrame, classifier, regressor):
    """Strona testów modeli"""
    st.markdown("## 🧪 Testy Poprawionych Modeli")
    
    if not classifier:
        st.error("❌ Brak klasyfikatora do testów")
        return
    
    st.markdown("### 📊 Informacje o Modelach")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Klasyfikator Europejski**")
        st.info(f"""
        - **Typ**: {classifier.model_type}
        - **Features**: {len(classifier.feature_names) if hasattr(classifier, 'feature_names') else 'N/A'}
        - **Status**: ✅ Poprawiony (bez data leakage)
        - **Class weight**: balanced
        """)
    
    with col2:
        if regressor:
            st.markdown("**📈 Regressor Europejski**")
            st.info(f"""
            - **Typ**: {regressor.model_type}
            - **Status**: ✅ Poprawiony
            - **Regularyzacja**: Tak
            - **Outliers**: Usunięte >4h
            """)
    
    # Feature importance
    if hasattr(classifier, 'get_feature_importance'):
        st.markdown("### 🏆 Najważniejsze Cechy (bez data leakage)")
        try:
            importance = classifier.get_feature_importance()
            if importance is not None:
                top_features = importance.head(15)
                
                fig = px.bar(top_features, x='importance', y='feature', 
                           title='🔍 Top 15 Najważniejszych Cech',
                           labels={'importance': 'Ważność', 'feature': 'Cecha'},
                           color='importance', color_continuous_scale='viridis')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Sprawdź czy nie ma data leakage
                suspicious_features = [f for f in top_features['feature'].tolist() 
                                     if any(word in f.lower() for word in ['delay', 'late', 'is_delayed'])]
                
                if suspicious_features:
                    st.error(f"⚠️ UWAGA: Podejrzane cechy: {suspicious_features}")
                else:
                    st.success("✅ Brak podejrzanych cech - poprawka udana!")
            else:
                st.warning("⚠️ Nie można wyświetlić feature importance")
        except Exception as e:
            st.error(f"❌ Błąd feature importance: {str(e)}")
    
    # Testy na losowych lotach
    st.markdown("### 🎲 Test na Losowych Lotach")
    
    if st.button("🎯 Testuj 5 losowych lotów"):
        sample_flights = data.sample(5)
        
        for idx, (_, flight) in enumerate(sample_flights.iterrows(), 1):
            st.markdown(f"**✈️ Test {idx}: {flight['airline']} {flight['origin']}→{flight['destination']}**")
            
            flight_details = {
                'flight_date': flight['flight_date'].strftime('%Y-%m-%d'),
                'airline': flight['airline'],
                'origin': flight['origin'],
                'destination': flight['destination'],
                'country_origin': flight['country_origin'],
                'country_destination': flight['country_destination'],
                'distance_km': flight['distance_km'],
                'scheduled_departure': flight['scheduled_departure'],
                'day_of_week': flight['flight_date'].weekday(),
                'month': flight['flight_date'].month,
                'hour': pd.to_datetime(flight['scheduled_departure'], format='%H:%M').hour
            }
            
            try:
                prediction = predict_european_delay(flight_details, classifier, regressor)
                actual_delayed = flight['delay_minutes'] > 15
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Przewidywanie", "OPÓŹNIONY" if prediction['is_delayed'] else "PUNKTUALNY")
                with col2:
                    st.metric("Rzeczywistość", "OPÓŹNIONY" if actual_delayed else "PUNKTUALNY")
                with col3:
                    st.metric("Prawdopodobieństwo", f"{prediction['delay_probability']*100:.1f}%")
                with col4:
                    st.metric("Rzeczywiste opóźnienie", f"{flight['delay_minutes']:.0f} min")
                
            except Exception as e:
                st.error(f"Błąd testu {idx}: {str(e)}")
            
            st.markdown("---")

if __name__ == "__main__":
    main() 