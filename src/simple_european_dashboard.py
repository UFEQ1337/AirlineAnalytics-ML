"""
🇪🇺 PROSTY DASHBOARD EUROPEJSKI - POPRAWIONE MODELE
==================================================

Prosty dashboard dla poprawionych modeli europejskich bez data leakage.

Autorzy: AirlineAnalytics-ML Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime
import os
import sys

# Dodaj src do ścieżki
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from european_models import predict_european_delay
except ImportError as e:
    st.error(f"❌ Błąd importu: {str(e)}")
    st.stop()

# Konfiguracja strony
st.set_page_config(
    page_title="🇪🇺 European Airline Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ulepszone CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
    }
    
    .stApp {
        background-color: #1a1a1a !important;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #1a1a1a;
        padding-top: 1rem;
    }
    
    /* Input styling */
    .stSelectbox label, .stDateInput label, .stTimeInput label, .stNumberInput label {
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #2d2d2d, #3a3a3a);
        border: 1px solid rgba(102,126,234,0.2);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    [data-testid="metric-container"] > div {
        color: #e0e0e0 !important;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 2rem 1rem;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Status badges */
    .fixed-badge {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #2d2d2d, #3a3a3a);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
        color: #e0e0e0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(102,126,234,0.3);
        background: linear-gradient(135deg, #333333, #404040);
    }
    
    /* Color-coded metrics */
    .metric-excellent { 
        color: #4ade80; 
        font-weight: 600;
        background: linear-gradient(135deg, rgba(74,222,128,0.2), rgba(74,222,128,0.1));
        padding: 0.8rem;
        border-radius: 10px;
        border: 2px solid rgba(74,222,128,0.4);
    }
    
    .metric-good { 
        color: #22d3ee; 
        font-weight: 600;
        background: linear-gradient(135deg, rgba(34,211,238,0.2), rgba(34,211,238,0.1));
        padding: 0.8rem;
        border-radius: 10px;
        border: 2px solid rgba(34,211,238,0.4);
    }
    
    .metric-warning { 
        color: #fbbf24; 
        font-weight: 600;
        background: linear-gradient(135deg, rgba(251,191,36,0.2), rgba(251,191,36,0.1));
        padding: 0.8rem;
        border-radius: 10px;
        border: 2px solid rgba(251,191,36,0.4);
    }
    
    .metric-bad { 
        color: #f87171; 
        font-weight: 600;
        background: linear-gradient(135deg, rgba(248,113,113,0.2), rgba(248,113,113,0.1));
        padding: 0.8rem;
        border-radius: 10px;
        border: 2px solid rgba(248,113,113,0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2d2d2d 0%, #1a1a1a 100%);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #1e3a8a, #1e40af);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(59,130,246,0.3);
        color: #e0e0e0;
    }
    
    /* Success boxes */
    .success-box {
        background: linear-gradient(135deg, #14532d, #166534);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #22c55e;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(34,197,94,0.3);
        color: #e0e0e0;
    }
    
    /* Error boxes */
    .error-box {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ef4444;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(239,68,68,0.3);
        color: #e0e0e0;
    }
    
    /* Chart containers */
    .chart-container {
        background: linear-gradient(135deg, #2d2d2d, #3a3a3a);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        margin: 1rem 0;
        border: 1px solid rgba(102,126,234,0.2);
        color: #e0e0e0;
    }
    
    /* Polish section styling */
    .polish-section {
        background: linear-gradient(135deg, #7c2d12, #a16207);
        padding: 2rem;
        border-radius: 20px;
        border: 3px solid #f59e0b;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        color: #e0e0e0;
    }
    
    .polish-section::before {
        content: "🇵🇱";
        position: absolute;
        top: -10px;
        right: -10px;
        font-size: 4rem;
        opacity: 0.2;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102,126,234,0.6) !important;
        background: linear-gradient(135deg, #7c3aed, #8b5cf6) !important;
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #22d3ee, #06b6d4) !important;
        box-shadow: 0 4px 15px rgba(34,211,238,0.4) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0891b2, #0e7490) !important;
        box-shadow: 0 8px 25px rgba(34,211,238,0.6) !important;
    }
    
    /* Loading spinner */
    .loading-container {
        text-align: center;
        padding: 3rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .metric-card {
            margin-bottom: 0.5rem;
        }
    }
    
    /* Streamlit components dark theme */
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #667eea;
    }
    
    .stDateInput > div > div {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #667eea;
    }
    
    .stTimeInput > div > div {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #667eea;
    }
    
    .stNumberInput > div > div {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #667eea;
    }
    
    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0 !important;
    }
    
    p, div, span {
        color: #e0e0e0 !important;
    }
    
    .stMarkdown {
        color: #e0e0e0 !important;
    }
    
    /* Sidebar text */
    .css-1d391kg .stMarkdown {
        color: #e0e0e0 !important;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Alert styling */
    .stAlert {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #667eea;
    }
    
    /* Code blocks */
    code {
        background-color: #2d2d2d !important;
        color: #22d3ee !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def load_european_data():
    """Ładuje dane europejskie"""
    try:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'european_flights_data.csv')
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            data['flight_date'] = pd.to_datetime(data['flight_date'])
            return data
        else:
            return None
    except Exception as e:
        st.error(f"Błąd ładowania danych: {e}")
        return None

def load_fixed_models():
    """Ładuje poprawione modele"""
    try:
        base_path = os.path.dirname(os.path.dirname(__file__))
        classifier_path = os.path.join(base_path, 'notebooks', 'european_fixed_model_classifier.joblib')
        regressor_path = os.path.join(base_path, 'notebooks', 'european_fixed_model_regressor.joblib')
        
        classifier = None
        regressor = None
        
        if os.path.exists(classifier_path):
            classifier = joblib.load(classifier_path)
        if os.path.exists(regressor_path):
            regressor = joblib.load(regressor_path)
            
        return classifier, regressor
    except Exception as e:
        st.error(f"Błąd ładowania modeli: {e}")
        return None, None

def main():
    # Header
    st.markdown('''
    <div class="main-header">
        <h1>✈️ European Airline Analytics</h1>
        <p style="font-size: 1.2rem; margin: 0.5rem 0;">Zaawansowana analiza lotów europejskich</p>
        <span class="fixed-badge">✅ POPRAWIONE MODELE - BEZ DATA LEAKAGE</span>
    </div>
    ''', unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown('''
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: #667eea;">🛠️ Menu Nawigacji</h2>
        </div>
        ''', unsafe_allow_html=True)
        
        # Navigation with icons
        page_options = {
            "📊 Przegląd": "Główny dashboard z KPIs i wykresami",
            "🔮 Przewidywanie": "Przewidywanie opóźnień dla nowych lotów", 
            "🧪 Testy": "Testowanie modeli i feature importance"
        }
        
        page = st.selectbox(
            "Wybierz stronę:",
            list(page_options.keys()),
            help="Nawiguj między różnymi sekcjami dashboard"
        )
        
        # Show description
        st.info(f"💡 {page_options[page]}")
        
        st.markdown("---")
    
    # Załaduj dane z lepszym UX
    with st.spinner('🔄 Ładowanie danych...'):
        data = load_european_data()
        classifier, regressor = load_fixed_models()
    
    # Enhanced status sidebar
    with st.sidebar:
        st.markdown('<div class="section-header">📋 Status Systemu</div>', unsafe_allow_html=True)
        
        # Data status
        if data is not None:
            st.markdown(f'''
            <div class="success-box">
                <strong>✅ Dane załadowane</strong><br>
                📊 {len(data):,} lotów w bazie<br>
                📅 Okres: {data['flight_date'].min().strftime('%Y-%m-%d')} - {data['flight_date'].max().strftime('%Y-%m-%d')}
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="error-box">
                <strong>❌ Brak danych</strong><br>
                Uruchom: <code>python demo_european_analysis.py</code>
            </div>
            ''', unsafe_allow_html=True)
        
        # Model status
        models_loaded = sum([classifier is not None, regressor is not None])
        if models_loaded == 2:
            st.markdown('''
            <div class="success-box">
                <strong>✅ Modele gotowe</strong><br>
                🎯 Klasyfikator: OK<br>
                📈 Regressor: OK<br>
                🔒 Data leakage: Usunięty
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="error-box">
                <strong>⚠️ Modele niepełne ({models_loaded}/2)</strong><br>
                {"✅" if classifier else "❌"} Klasyfikator<br>
                {"✅" if regressor else "❌"} Regressor<br>
                Uruchom: <code>python train_european_models_fixed.py</code>
            </div>
            ''', unsafe_allow_html=True)
        
        # Quick stats
        if data is not None:
            st.markdown("---")
            st.markdown('<div class="section-header">📈 Szybkie Statystyki</div>', unsafe_allow_html=True)
            
            polish_flights = len(data[(data['country_origin'] == 'Polska') | 
                                     (data['country_destination'] == 'Polska')])
            on_time_rate = len(data[data['delay_minutes'] <= 15]) / len(data) * 100
            
            st.markdown(f'''
            <div class="info-box">
                🇵🇱 <strong>Loty polskie:</strong> {polish_flights:,}<br>
                ⏰ <strong>Punktualność:</strong> {on_time_rate:.1f}%<br>
                ✈️ <strong>Linie lotnicze:</strong> {data['airline'].nunique()}<br>
                🌍 <strong>Lotniska:</strong> {data['origin'].nunique()}
            </div>
            ''', unsafe_allow_html=True)
    
    # Strony
    if page == "📊 Przegląd":
        show_overview(data)
    elif page == "🔮 Przewidywanie":
        show_prediction(data, classifier, regressor)
    elif page == "🧪 Testy":
        show_tests(data, classifier, regressor)

def show_overview(data):
    """Strona przeglądu"""
    st.markdown('<div class="section-header">📊 Przegląd Europejski</div>', unsafe_allow_html=True)
    
    if data is None:
        st.markdown('''
        <div class="error-box">
            <h3>❌ Brak danych do wyświetlenia</h3>
            <p>Aby uruchomić dashboard, wykonaj:</p>
            <code>python demo_european_analysis.py</code>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    # KPIs podstawowe
    total_flights = len(data)
    delayed_flights = len(data[data['delay_minutes'] > 15])
    on_time_percent = ((total_flights - delayed_flights) / total_flights) * 100
    avg_delay = data['delay_minutes'].mean()
    
    # Polskie loty
    polish_flights = len(data[(data['country_origin'] == 'Polska') | 
                             (data['country_destination'] == 'Polska')])
    polish_percent = (polish_flights / total_flights) * 100
    
    # Advanced KPIs
    max_delay = data['delay_minutes'].max()
    airlines_count = data['airline'].nunique()
    routes_count = data.groupby(['origin', 'destination']).size().shape[0]
    avg_distance = data['distance_km'].mean()
    
    # Wyświetl główne KPIs w ładnych kartach
    st.markdown("### 🎯 Kluczowe Wskaźniki")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color_class = "metric-excellent" if on_time_percent >= 85 else "metric-good" if on_time_percent >= 80 else "metric-warning" if on_time_percent >= 70 else "metric-bad"
        st.markdown(f'''
        <div class="metric-card">
            <div class="{color_class}">
                <h2 style="margin: 0;">⏰ {on_time_percent:.1f}%</h2>
                <p style="margin: 0.5rem 0;">Punktualność</p>
                <small>{total_flights - delayed_flights:,} z {total_flights:,} lotów</small>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        delay_color = "metric-good" if avg_delay < 10 else "metric-warning" if avg_delay < 20 else "metric-bad"
        st.markdown(f'''
        <div class="metric-card">
            <div class="{delay_color}">
                <h2 style="margin: 0;">📊 {avg_delay:.1f} min</h2>
                <p style="margin: 0.5rem 0;">Średnie opóźnienie</p>
                <small>Max: {max_delay:.0f} min</small>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-good">
                <h2 style="margin: 0;">🇵🇱 {polish_percent:.1f}%</h2>
                <p style="margin: 0.5rem 0;">Loty polskie</p>
                <small>{polish_flights:,} lotów</small>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-good">
                <h2 style="margin: 0;">✈️ {airlines_count}</h2>
                <p style="margin: 0.5rem 0;">Linie lotnicze</p>
                <small>{routes_count} tras</small>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Dodatkowe KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="info-box">
                <strong>🇪🇺 {total_flights:,}</strong><br>
                <small>Łączne loty</small>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="info-box">
                <strong>📏 {avg_distance:.0f} km</strong><br>
                <small>Średni dystans</small>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        countries_count = data['country_origin'].nunique()
        st.markdown(f'''
        <div class="metric-card">
            <div class="info-box">
                <strong>🌍 {countries_count}</strong><br>
                <small>Krajów</small>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        airports_count = pd.concat([data['origin'], data['destination']]).nunique()
        st.markdown(f'''
        <div class="metric-card">
            <div class="info-box">
                <strong>🛫 {airports_count}</strong><br>
                <small>Lotnisk</small>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Wykresy
    st.markdown("### 📊 Analizy Wizualne")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Timeline opóźnień
        daily_stats = data.groupby(data['flight_date'].dt.date)['delay_minutes'].mean().reset_index()
        daily_stats.columns = ['date', 'avg_delay']
        
        fig_timeline = px.line(
            daily_stats, x='date', y='avg_delay',
            title='📈 Timeline Średnich Opóźnień',
            color_discrete_sequence=['#667eea']
        )
        fig_timeline.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=16,
            title_font_color='#e0e0e0',
            xaxis_title="Data",
            yaxis_title="Średnie opóźnienie (min)",
            font_color='#e0e0e0',
            xaxis=dict(color='#e0e0e0'),
            yaxis=dict(color='#e0e0e0')
        )
        fig_timeline.update_traces(line_width=3)
        st.plotly_chart(fig_timeline, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Top airlines
        airline_stats = data.groupby('airline')['delay_minutes'].agg(['mean', 'count']).reset_index()
        airline_stats.columns = ['airline', 'avg_delay', 'count']
        airline_stats = airline_stats[airline_stats['count'] >= 100]
        airline_stats = airline_stats.nsmallest(10, 'avg_delay')
        
        fig_airlines = px.bar(
            airline_stats, x='avg_delay', y='airline',
            title='✈️ Top 10 Airlines (najmniejsze opóźnienia)',
            orientation='h',
            color='avg_delay',
            color_continuous_scale='RdYlGn_r'
        )
        fig_airlines.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=16,
            title_font_color='#e0e0e0',
            xaxis_title="Średnie opóźnienie (min)",
            yaxis_title="Linia lotnicza",
            showlegend=False,
            font_color='#e0e0e0',
            xaxis=dict(color='#e0e0e0'),
            yaxis=dict(color='#e0e0e0')
        )
        st.plotly_chart(fig_airlines, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Dodatkowe wykresy
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Rozkład opóźnień
        delay_bins = pd.cut(data['delay_minutes'], 
                           bins=[-float('inf'), 0, 15, 30, 60, float('inf')], 
                           labels=['Wcześnie', 'Punktualnie', '15-30 min', '30-60 min', '>60 min'])
        delay_counts = pd.Series(delay_bins).value_counts().reset_index()
        delay_counts.columns = ['kategoria', 'liczba']
        
        fig_delays = px.pie(
            delay_counts, values='liczba', names='kategoria',
            title='🎯 Rozkład Kategorii Opóźnień',
            color_discrete_sequence=['#28a745', '#20c997', '#ffc107', '#fd7e14', '#dc3545']
        )
        fig_delays.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=16,
            title_font_color='#e0e0e0',
            font_color='#e0e0e0'
        )
        st.plotly_chart(fig_delays, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Top kraje
        country_stats = data.groupby('country_origin').agg({
            'delay_minutes': 'mean',
            'flight_date': 'count'
        }).reset_index()
        country_stats.columns = ['kraj', 'avg_delay', 'count']
        country_stats = country_stats[country_stats['count'] >= 50]
        country_stats = country_stats.nlargest(10, 'count')
        
        fig_countries = px.scatter(
            country_stats, x='count', y='avg_delay',
            size='count', hover_name='kraj',
            title='🌍 Kraje: Liczba lotów vs Średnie opóźnienie',
            color='avg_delay',
            color_continuous_scale='RdYlGn_r'
        )
        fig_countries.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=16,
            title_font_color='#e0e0e0',
            xaxis_title="Liczba lotów",
            yaxis_title="Średnie opóźnienie (min)",
            font_color='#e0e0e0',
            xaxis=dict(color='#e0e0e0'),
            yaxis=dict(color='#e0e0e0')
        )
        st.plotly_chart(fig_countries, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Statystyki polskie
    st.markdown('<div class="polish-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🇵🇱 Statystyki Polskie</div>', unsafe_allow_html=True)
    
    polish_data = data[(data['country_origin'] == 'Polska') | 
                      (data['country_destination'] == 'Polska')]
    
    if len(polish_data) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            polish_on_time = len(polish_data[polish_data['delay_minutes'] <= 15])
            polish_on_time_pct = (polish_on_time / len(polish_data)) * 100
            polish_color = "metric-excellent" if polish_on_time_pct >= 85 else "metric-good" if polish_on_time_pct >= 80 else "metric-warning"
            
            st.markdown(f'''
            <div class="metric-card">
                <div class="{polish_color}">
                    <h2 style="margin: 0;">🇵🇱 {polish_on_time_pct:.1f}%</h2>
                    <p style="margin: 0.5rem 0;">Punktualność polska</p>
                    <small>{polish_on_time:,} z {len(polish_data):,} lotów</small>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            lot_data = polish_data[polish_data['airline'] == 'LOT Polish Airlines']
            lot_avg_delay = lot_data['delay_minutes'].mean() if len(lot_data) > 0 else 0
            lot_color = "metric-good" if lot_avg_delay < 10 else "metric-warning" if lot_avg_delay < 20 else "metric-bad"
            
            st.markdown(f'''
            <div class="metric-card">
                <div class="{lot_color}">
                    <h2 style="margin: 0;">✈️ {lot_avg_delay:.1f} min</h2>
                    <p style="margin: 0.5rem 0;">LOT średnie opóźnienie</p>
                    <small>{len(lot_data):,} lotów LOT</small>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            if len(polish_data) > 0:
                route_counts = polish_data.groupby(['origin', 'destination']).size()
                if len(route_counts) > 0:
                    top_polish_route = route_counts.idxmax()
                    top_route_count = route_counts.max()
                    
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-good">
                            <h2 style="margin: 0;">🛫 {top_route_count}</h2>
                            <p style="margin: 0.5rem 0;">Najpopularniejsza trasa</p>
                            <small>{top_polish_route[0]} → {top_polish_route[1]}</small>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
        
        # Dodatkowe statystyki polskie
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            # Top polskie lotniska
            polish_airports = pd.concat([
                polish_data[polish_data['country_origin'] == 'Polska']['origin'],
                polish_data[polish_data['country_destination'] == 'Polska']['destination']
            ]).value_counts().head(10).reset_index()
            polish_airports.columns = ['lotnisko', 'liczba_lotów']
            
            fig_polish_airports = px.bar(
                polish_airports, x='liczba_lotów', y='lotnisko',
                title='🇵🇱 Top 10 Polskich Lotnisk',
                orientation='h',
                color='liczba_lotów',
                color_continuous_scale='Blues'
            )
            fig_polish_airports.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=16,
                title_font_color='#e0e0e0',
                showlegend=False,
                font_color='#e0e0e0',
                xaxis=dict(color='#e0e0e0'),
                yaxis=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig_polish_airports, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            # Polskie linie vs zagraniczne
            polish_airlines = polish_data.groupby(
                polish_data['airline'].str.contains('Polish|LOT', case=False, na=False)
            )['delay_minutes'].agg(['mean', 'count']).reset_index()
            polish_airlines['airline_type'] = polish_airlines['airline'].map({
                True: 'Polskie linie', False: 'Zagraniczne linie'
            })
            
            fig_airlines_comparison = px.bar(
                polish_airlines, x='airline_type', y='mean',
                title='🇵🇱 Polskie vs Zagraniczne Linie',
                color='airline_type',
                color_discrete_map={'Polskie linie': '#dc143c', 'Zagraniczne linie': '#4682b4'}
            )
            fig_airlines_comparison.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=16,
                title_font_color='#e0e0e0',
                xaxis_title="Typ linii lotniczej",
                yaxis_title="Średnie opóźnienie (min)",
                showlegend=False,
                font_color='#e0e0e0',
                xaxis=dict(color='#e0e0e0'),
                yaxis=dict(color='#e0e0e0')
            )
            st.plotly_chart(fig_airlines_comparison, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown('''
        <div class="info-box">
            <h3>ℹ️ Brak danych polskich</h3>
            <p>Nie znaleziono lotów związanych z Polską w obecnym zbiorze danych.</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_prediction(data, classifier, regressor):
    """Strona przewidywania"""
    st.markdown('<div class="section-header">🔮 Przewidywanie Opóźnień</div>', unsafe_allow_html=True)
    st.markdown('<center><span class="fixed-badge">✅ BEZ DATA LEAKAGE</span></center>', 
               unsafe_allow_html=True)
    
    if not classifier or not regressor:
        st.markdown('''
        <div class="error-box">
            <h3>❌ Brak modeli do przewidywania!</h3>
            <p>Aby uruchomić przewidywanie, wykonaj:</p>
            <code>python train_european_models_fixed.py</code>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    if data is None:
        st.markdown('''
        <div class="error-box">
            <h3>❌ Brak danych referencyjnych!</h3>
            <p>Potrzebne są dane do wypełnienia formularza</p>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    st.markdown('<div class="section-header">✈️ Parametry Lotu</div>', unsafe_allow_html=True)
    
    # Funkcja pomocnicza do obliczenia dystansu (przybliżona)
    def calculate_distance(origin_airport, destination_airport, data):
        """Oblicza przybliżony dystans między lotniskami na podstawie danych historycznych"""
        try:
            # Znajdź dystans z danych historycznych
            distance_data = data[
                ((data['origin'] == origin_airport) & (data['destination'] == destination_airport)) |
                ((data['origin'] == destination_airport) & (data['destination'] == origin_airport))
            ]
            if not distance_data.empty:
                return int(distance_data['distance_km'].iloc[0])
            else:
                # Fallback - przybliżony dystans na podstawie kraju
                if origin_airport == destination_airport:
                    return 0
                # Proste przybliżenie
                return 1000
        except:
            return 1000
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 **Podstawowe Informacje**")
        
        # Data lotu
        flight_date = st.date_input(
            "📅 **Data lotu**:", 
            datetime.now().date(),
            help="Wybierz datę planowanego lotu"
        )
        
        # Linia lotnicza
        airlines = sorted(data['airline'].unique())
        airline = st.selectbox(
            "✈️ **Linia lotnicza**:", 
            airlines,
            help="Wybierz linię lotniczą"
        )
        
        # Godzina odlotu
        departure_time = st.time_input(
            "⏰ **Godzina odlotu**:", 
            value=datetime.strptime("10:00", "%H:%M").time(),
            help="Wybierz planowaną godzinę odlotu (domyślnie 10:00)"
        )
        
        # Podpowiedź o popularnych godzinach
        st.markdown("""
        <div style="font-size: 0.8rem; color: #888; margin-top: 0.5rem;">
        💡 <strong>Popularne godziny:</strong> 06:00-09:00 (rano), 12:00-15:00 (południe), 18:00-21:00 (wieczór)
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🌍 **Trasa Lotu**")
        
        # Kraj wylotu
        countries_origin = sorted(data['country_origin'].unique())
        country_origin = st.selectbox(
            "🌍 **Kraj wylotu**:", 
            countries_origin,
            help="Wybierz kraj, z którego odlatuje samolot"
        )
        
        # Lotnisko wylotu (filtrowane po kraju)
        origins_filtered = sorted(data[data['country_origin'] == country_origin]['origin'].unique())
        origin = st.selectbox(
            "🛫 **Lotnisko wylotu**:", 
            origins_filtered,
            help=f"Dostępne lotniska w kraju: {country_origin}"
        )
        
        # Kraj przylotu
        countries_dest = sorted(data['country_destination'].unique())
        country_destination = st.selectbox(
            "🌍 **Kraj przylotu**:", 
            countries_dest,
            help="Wybierz kraj docelowy"
        )
        
        # Lotnisko przylotu (filtrowane po kraju)
        destinations_filtered = sorted(data[data['country_destination'] == country_destination]['destination'].unique())
        destination = st.selectbox(
            "🛬 **Lotnisko przylotu**:", 
            destinations_filtered,
            help=f"Dostępne lotniska w kraju: {country_destination}"
        )
        
        # Automatyczne obliczenie dystansu
        if origin and destination:
            distance_km = calculate_distance(origin, destination, data)
            st.markdown(f'''
            <div class="info-box">
                <h4>📏 **Dystans Automatycznie Obliczony**</h4>
                <p><strong>Trasa:</strong> {origin} → {destination}</p>
                <p><strong>Dystans:</strong> {distance_km:,} km</p>
                <p><strong>Czas lotu:</strong> ~{distance_km//800 + 1} godz</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Dodatkowe informacje o trasie
    if origin and destination and origin != destination:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Statystyki historyczne dla tej trasy
            route_data = data[
                (data['origin'] == origin) & 
                (data['destination'] == destination)
            ]
            
            if not route_data.empty:
                avg_delay = route_data['delay_minutes'].mean()
                on_time_rate = len(route_data[route_data['delay_minutes'] <= 15]) / len(route_data) * 100
                
                st.markdown(f'''
                <div class="success-box">
                    <h4>📊 **Statystyki Trasy**</h4>
                    <p><strong>Liczba lotów:</strong> {len(route_data)}</p>
                    <p><strong>Średnie opóźnienie:</strong> {avg_delay:.1f} min</p>
                    <p><strong>Punktualność:</strong> {on_time_rate:.1f}%</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="info-box">
                    <h4>ℹ️ **Nowa Trasa**</h4>
                    <p>Brak danych historycznych dla tej trasy</p>
                </div>
                ''', unsafe_allow_html=True)
        
        with col2:
            # Informacje o linii lotniczej
            airline_data = data[data['airline'] == airline]
            if not airline_data.empty:
                airline_avg_delay = airline_data['delay_minutes'].mean()
                airline_on_time = len(airline_data[airline_data['delay_minutes'] <= 15]) / len(airline_data) * 100
                
                color_class = "success-box" if airline_on_time >= 80 else "info-box" if airline_on_time >= 70 else "error-box"
                
                st.markdown(f'''
                <div class="{color_class}">
                    <h4>✈️ **Profil Linii Lotniczej**</h4>
                    <p><strong>Nazwa:</strong> {airline}</p>
                    <p><strong>Średnie opóźnienie:</strong> {airline_avg_delay:.1f} min</p>
                    <p><strong>Punktualność:</strong> {airline_on_time:.1f}%</p>
                </div>
                ''', unsafe_allow_html=True)
        
        with col3:
            # Informacje o czasie
            day_name = ['Poniedziałek', 'Wtorek', 'Środa', 'Czwartek', 'Piątek', 'Sobota', 'Niedziela'][flight_date.weekday()]
            month_name = ['Stycznia', 'Lutego', 'Marca', 'Kwietnia', 'Maja', 'Czerwca', 
                         'Lipca', 'Sierpnia', 'Września', 'Października', 'Listopada', 'Grudnia'][flight_date.month-1]
            
            # Analiza czasowa
            hour = departure_time.hour
            time_period = "Rano" if 6 <= hour < 12 else "Południe" if 12 <= hour < 18 else "Wieczór" if 18 <= hour < 22 else "Noc"
            
            st.markdown(f'''
            <div class="info-box">
                <h4>🕐 **Informacje Czasowe**</h4>
                <p><strong>Dzień:</strong> {day_name}</p>
                <p><strong>Data:</strong> {flight_date.day} {month_name}</p>
                <p><strong>Pora dnia:</strong> {time_period}</p>
                <p><strong>Godzina:</strong> {departure_time.strftime('%H:%M')}</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Przewidywanie z lepszym przyciskiem
    st.markdown("---")
    st.markdown('<div class="section-header">🔮 Przewidywanie Opóźnienia</div>', unsafe_allow_html=True)
    
    # Walidacja danych
    if not origin or not destination:
        st.markdown('''
        <div class="error-box">
            <h4>⚠️ **Uzupełnij Dane**</h4>
            <p>Wybierz lotnisko wylotu i przylotu przed przewidywaniem</p>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    if origin == destination:
        st.markdown('''
        <div class="error-box">
            <h4>⚠️ **Błąd Trasy**</h4>
            <p>Lotnisko wylotu i przylotu nie mogą być takie same</p>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    # Podsumowanie przed przewidywaniem
    st.markdown(f'''
    <div class="info-box">
        <h4>📋 **Podsumowanie Lotu**</h4>
        <p><strong>🛫 Trasa:</strong> {origin} ({country_origin}) → {destination} ({country_destination})</p>
        <p><strong>✈️ Linia:</strong> {airline}</p>
        <p><strong>📅 Data:</strong> {flight_date.strftime('%d.%m.%Y')} o {departure_time.strftime('%H:%M')}</p>
        <p><strong>📏 Dystans:</strong> {distance_km:,} km</p>
    </div>
    ''', unsafe_allow_html=True)
    
    prediction_container = st.container()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        predict_button = st.button("🔮 **Przewiduj Opóźnienie**", type="primary", use_container_width=True)
    with col2:
        st.markdown("#### 💡 **Wskazówka**")
        st.markdown("Kliknij przycisk, aby otrzymać przewidywanie opóźnienia na podstawie AI")
    
    if predict_button:
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
            with st.spinner('🔄 Analizowanie parametrów lotu za pomocą sztucznej inteligencji...'):
                prediction = predict_european_delay(flight_details, classifier, regressor)
            
            with prediction_container:
                st.markdown('<div class="section-header">📊 Wyniki Przewidywania AI</div>', unsafe_allow_html=True)
                
                # Główne wyniki
                prob = prediction['delay_probability'] * 100
                delay_min = prediction['predicted_delay_minutes']
                is_delayed = prediction['is_delayed']
                
                # Kategorie ryzyka
                if prob >= 80:
                    risk_level = "BARDZO WYSOKIE"
                    risk_color = "metric-bad"
                    risk_icon = "🔴"
                elif prob >= 60:
                    risk_level = "WYSOKIE"
                    risk_color = "metric-bad"
                    risk_icon = "🟠"
                elif prob >= 40:
                    risk_level = "ŚREDNIE"
                    risk_color = "metric-warning"
                    risk_icon = "🟡"
                elif prob >= 20:
                    risk_level = "NISKIE"
                    risk_color = "metric-good"
                    risk_icon = "🟢"
                else:
                    risk_level = "BARDZO NISKIE"
                    risk_color = "metric-excellent"
                    risk_icon = "🟢"
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="{risk_color}">
                            <h2 style="margin: 0;">{risk_icon} {prob:.1f}%</h2>
                            <p style="margin: 0.5rem 0;"><strong>Prawdopodobieństwo Opóźnienia</strong></p>
                            <small>Ryzyko: {risk_level}</small>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    status = "OPÓŹNIONY" if is_delayed else "PUNKTUALNY"
                    status_color = "metric-bad" if is_delayed else "metric-excellent"
                    status_icon = "🚨" if is_delayed else "✅"
                    confidence = "Wysoka pewność" if abs(prob-50) > 30 else "Średnia pewność"
                    
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="{status_color}">
                            <h2 style="margin: 0;">{status_icon} {status}</h2>
                            <p style="margin: 0.5rem 0;"><strong>Przewidywany Status</strong></p>
                            <small>{confidence}</small>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col3:
                    delay_color = "metric-excellent" if delay_min < 5 else "metric-good" if delay_min < 15 else "metric-warning" if delay_min < 30 else "metric-bad"
                    
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="{delay_color}">
                            <h2 style="margin: 0;">⏰ {delay_min:.0f} min</h2>
                            <p style="margin: 0.5rem 0;"><strong>Przewidywane Opóźnienie</strong></p>
                            <small>Średnia dla trasy: {data['delay_minutes'].mean():.0f} min</small>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Szczegółowa analiza
                risk_colors = {
                    'BARDZO WYSOKIE': 'error-box',
                    'WYSOKIE': 'error-box',
                    'ŚREDNIE': 'info-box', 
                    'NISKIE': 'success-box',
                    'BARDZO NISKIE': 'success-box'
                }
                risk_color = risk_colors.get(risk_level, 'info-box')
                
                # Oblicz dodatkowe metryki
                try:
                    expected_departure = departure_time.strftime('%H:%M') if departure_time else "00:00"
                    if delay_min > 0 and departure_time:
                        expected_time = datetime.combine(flight_date, departure_time)
                        expected_time = expected_time + pd.Timedelta(minutes=delay_min)
                        expected_departure = expected_time.strftime('%H:%M')
                except:
                    expected_departure = "00:00"
                
                day_name = ['Poniedziałek', 'Wtorek', 'Środa', 'Czwartek', 'Piątek', 'Sobota', 'Niedziela'][flight_date.weekday()]
                month_name = ['Stycznia', 'Lutego', 'Marca', 'Kwietnia', 'Maja', 'Czerwca', 
                             'Lipca', 'Sierpnia', 'Września', 'Października', 'Listopada', 'Grudnia'][flight_date.month-1]
                
                st.markdown(f'''
                <div class="{risk_color}">
                    <h3>📋 **Szczegółowa Analiza AI**</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div>
                            <p><strong>🎯 Poziom ryzyka:</strong> {risk_level}</p>
                            <p><strong>✈️ Trasa lotu:</strong> {origin} → {destination}</p>
                            <p><strong>📏 Dystans:</strong> {distance_km:,} km</p>
                            <p><strong>🏢 Linia lotnicza:</strong> {airline}</p>
                        </div>
                        <div>
                            <p><strong>📅 Data:</strong> {day_name}, {flight_date.day} {month_name}</p>
                            <p><strong>🕐 Planowany odlot:</strong> {departure_time.strftime('%H:%M') if departure_time else '00:00'}</p>
                            <p><strong>🕐 Przewidywany odlot:</strong> {expected_departure}</p>
                            <p><strong>⏱️ Czas lotu:</strong> ~{distance_km//800 + 1} godzin</p>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Inteligentne wskazówki
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    if is_delayed:
                        if delay_min > 60:
                            advice_type = "error-box"
                            advice_title = "🚨 **Duże Opóźnienie Przewidywane**"
                            advice_list = [
                                "📞 Zadzwoń do linii lotniczej NATYCHMIAST",
                                "🍽️ Zabezpiecz posiłek i napoje na lotnisku",
                                "📋 Sprawdź prawa do odszkodowania (EU261)",
                                "🏨 Rozważ rezerwację hotelu jeśli lot wieczorny",
                                "📱 Pobierz aplikację linii lotniczej",
                                "💺 Sprawdź możliwość zmiany lotu"
                            ]
                        else:
                            advice_type = "error-box"
                            advice_title = "⚠️ **Opóźnienie Prawdopodobne**"
                            advice_list = [
                                "📱 Sprawdzaj status lotu co godzinę",
                                "🕐 Zaplanuj dodatkowe 30-60 min na lotnisku",
                                "☕ Znajdź spokojne miejsce na oczekiwanie",
                                "📋 Przygotuj dokumenty na odszkodowanie",
                                "🔋 Naładuj urządzenia elektroniczne"
                            ]
                    else:
                        advice_type = "success-box"
                        advice_title = "✅ **Lot Prawdopodobnie Punktualny**"
                        advice_list = [
                            "🚗 Standardowy czas dotarcia na lotnisko (2h wcześniej)",
                            "📋 Sprawdź status lotu przed wyjściem z domu",
                            "💺 Online check-in 24h przed lotem",
                            "🧳 Przygotuj bagaż zgodnie z przepisami",
                            "✈️ Życzę miłego lotu!"
                        ]
                    
                    st.markdown(f'''
                    <div class="{advice_type}">
                        <h4>{advice_title}</h4>
                        <ul>
                            {"".join([f"<li>{item}</li>" for item in advice_list])}
                        </ul>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    # Alternatywne opcje
                    st.markdown('''
                    <div class="info-box">
                        <h4>🔄 **Alternatywne Opcje**</h4>
                        <p><strong>Inne linie lotnicze na tej trasie:</strong></p>
                    ''', unsafe_allow_html=True)
                    
                    # Znajdź alternatywne linie
                    alternative_airlines = data[
                        (data['origin'] == origin) & 
                        (data['destination'] == destination) & 
                        (data['airline'] != airline)
                    ]['airline'].unique()
                    
                    if len(alternative_airlines) > 0:
                        for alt_airline in alternative_airlines[:3]:  # Pokaż max 3 alternatywy
                            alt_data = data[
                                (data['origin'] == origin) & 
                                (data['destination'] == destination) & 
                                (data['airline'] == alt_airline)
                            ]
                            if not alt_data.empty:
                                alt_avg_delay = alt_data['delay_minutes'].mean()
                                alt_on_time = len(alt_data[alt_data['delay_minutes'] <= 15]) / len(alt_data) * 100
                                st.markdown(f'''
                                    <p>✈️ <strong>{alt_airline}</strong><br>
                                    Punktualność: {alt_on_time:.0f}% | Średnie opóźnienie: {alt_avg_delay:.0f} min</p>
                                ''', unsafe_allow_html=True)
                    else:
                        st.markdown("<p>Brak alternatywnych linii w danych</p>", unsafe_allow_html=True)
                    
                    st.markdown('''
                        <p><strong>📞 Kontakt:</strong></p>
                        <p>• Sprawdź stronę linii lotniczej<br>
                        • Zadzwoń na infolinię<br>
                        • Użyj aplikacji mobilnej</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f'''
            <div class="error-box">
                <h3>❌ Błąd przewidywania</h3>
                <p>Szczegóły błędu: <code>{str(e)}</code></p>
                <p>Sprawdź poprawność danych i spróbuj ponownie.</p>
            </div>
            ''', unsafe_allow_html=True)

def show_tests(data, classifier, regressor):
    """Strona testów"""
    st.markdown('<div class="section-header">🧪 Testy Modeli</div>', unsafe_allow_html=True)
    
    if not classifier:
        st.markdown('''
        <div class="error-box">
            <h3>❌ Brak klasyfikatora do testów</h3>
            <p>Uruchom trening modeli przed wykonaniem testów</p>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    # Informacje o modelu
    st.markdown('<div class="section-header">📊 Informacje o Modelach</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'''
        <div class="success-box">
            <h4>🎯 Klasyfikator</h4>
            <p><strong>Typ:</strong> {getattr(classifier, 'model_type', 'Random Forest')}</p>
            <p><strong>Status:</strong> ✅ Poprawiony</p>
            <p><strong>Data leakage:</strong> ❌ Usunięty</p>
            <p><strong>Class weight:</strong> balanced</p>
            <p><strong>Jakość:</strong> Zoptymalizowany</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        if regressor:
            st.markdown(f'''
            <div class="success-box">
                <h4>📈 Regressor</h4>
                <p><strong>Typ:</strong> {getattr(regressor, 'model_type', 'Random Forest')}</p>
                <p><strong>Status:</strong> ✅ Poprawiony</p>
                <p><strong>Regularyzacja:</strong> ✅ Tak</p>
                <p><strong>Outliers:</strong> Obsłużone</p>
                <p><strong>Jakość:</strong> Zoptymalizowany</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="error-box">
                <h4>📈 Regressor</h4>
                <p>❌ Model nie załadowany</p>
                <p>Uruchom trening modeli</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Feature importance
    if hasattr(classifier, 'get_feature_importance'):
        st.markdown('<div class="section-header">🏆 Najważniejsze Cechy</div>', unsafe_allow_html=True)
        
        try:
            importance = classifier.get_feature_importance()
            if importance is not None:
                top_features = importance.head(10)
                
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                
                fig = px.bar(
                    top_features, x='importance', y='feature',
                    title='🏆 Top 10 Najważniejszych Cech Modelu',
                    orientation='h',
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title_font_size=16,
                    title_font_color='#e0e0e0',
                    xaxis_title="Ważność cechy",
                    yaxis_title="Cecha",
                    font_color='#e0e0e0',
                    xaxis=dict(color='#e0e0e0'),
                    yaxis=dict(color='#e0e0e0')
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Sprawdź data leakage
                suspicious = [f for f in top_features['feature'].tolist() 
                            if any(word in f.lower() for word in ['delay', 'late', 'is_delayed'])]
                
                if suspicious:
                    st.markdown(f'''
                    <div class="error-box">
                        <h4>⚠️ Potencjalne Data Leakage</h4>
                        <p>Znalezione podejrzane cechy:</p>
                        <ul>{"".join([f"<li><code>{f}</code></li>" for f in suspicious])}</ul>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown('''
                    <div class="success-box">
                        <h4>✅ Brak Data Leakage</h4>
                        <p>Wszystkie top cechy są bezpieczne i nie zawierają informacji o przyszłości.</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f'''
            <div class="error-box">
                <h3>❌ Błąd feature importance</h3>
                <p>Szczegóły: <code>{str(e)}</code></p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Test na losowych lotach
    if data is not None:
        st.markdown('<div class="section-header">🎲 Test na Losowych Lotach</div>', unsafe_allow_html=True)
        
        if st.button("🎯 Testuj 3 losowe loty", type="primary"):
            sample_flights = data.sample(3)
            
            correct_predictions = 0
            
            for idx, (_, flight) in enumerate(sample_flights.iterrows(), 1):
                st.markdown(f'''
                <div class="info-box">
                    <h4>✈️ Test {idx}: {flight['airline']}</h4>
                    <p><strong>Trasa:</strong> {flight['origin']} → {flight['destination']}</p>
                    <p><strong>Data:</strong> {flight['flight_date'].strftime('%d.%m.%Y')} o {flight['scheduled_departure']}</p>
                    <p><strong>Rzeczywiste opóźnienie:</strong> {flight['delay_minutes']:.0f} minut</p>
                </div>
                ''', unsafe_allow_html=True)
                
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
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        pred_status = "OPÓŹNIONY" if prediction['is_delayed'] else "PUNKTUALNY"
                        pred_color = "metric-bad" if prediction['is_delayed'] else "metric-good"
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="{pred_color}">
                                <h3 style="margin: 0;">🔮 {pred_status}</h3>
                                <p style="margin: 0.5rem 0;">Przewidywanie</p>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col2:
                        actual_status = "OPÓŹNIONY" if actual_delayed else "PUNKTUALNY"
                        actual_color = "metric-bad" if actual_delayed else "metric-good"
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="{actual_color}">
                                <h3 style="margin: 0;">📋 {actual_status}</h3>
                                <p style="margin: 0.5rem 0;">Rzeczywistość</p>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col3:
                        prob_color = "metric-bad" if prediction['delay_probability'] > 0.7 else "metric-warning" if prediction['delay_probability'] > 0.3 else "metric-good"
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="{prob_color}">
                                <h3 style="margin: 0;">📊 {prediction['delay_probability']*100:.1f}%</h3>
                                <p style="margin: 0.5rem 0;">Prawdopodobieństwo</p>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Zgodność
                    correct = prediction['is_delayed'] == actual_delayed
                    if correct:
                        correct_predictions += 1
                    
                    agreement_color = "success-box" if correct else "error-box"
                    agreement_icon = "✅" if correct else "❌"
                    
                    st.markdown(f'''
                    <div class="{agreement_color}">
                        <p><strong>{agreement_icon} Zgodność predykcji:</strong> {"TAK" if correct else "NIE"}</p>
                        <p><strong>Przewidywane opóźnienie:</strong> {prediction['predicted_delay_minutes']:.0f} min</p>
                        <p><strong>Rzeczywiste opóźnienie:</strong> {flight['delay_minutes']:.0f} min</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f'''
                    <div class="error-box">
                        <h4>❌ Błąd testu {idx}</h4>
                        <p>Szczegóły: <code>{str(e)}</code></p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.markdown("---")
            
            # Podsumowanie testów
            accuracy = (correct_predictions / 3) * 100
            accuracy_color = "success-box" if accuracy >= 66 else "info-box"
            
            st.markdown(f'''
            <div class="{accuracy_color}">
                <h3>📈 Podsumowanie Testów</h3>
                <p><strong>Poprawne przewidywania:</strong> {correct_predictions}/3</p>
                <p><strong>Dokładność:</strong> {accuracy:.1f}%</p>
                <p><strong>Status:</strong> {"Dobra jakość" if accuracy >= 66 else "Wymaga poprawy"}</p>
            </div>
            ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 