#!/usr/bin/env python3
"""
🇪🇺 URUCHOMIENIE DASHBOARD EUROPEJSKIEGO
=======================================

Prosty skrypt do uruchamiania dashboard dla poprawionych modeli europejskich.
Uruchom: streamlit run run_european_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from datetime import datetime
import os
import sys

# Dodaj src do ścieżki
sys.path.append('src')

try:
    from src.european_models import predict_european_delay
except ImportError:
    st.error("❌ Błąd importu modułów europejskich")
    st.stop()

# Konfiguracja
st.set_page_config(
    page_title="🇪🇺 European Dashboard",
    page_icon="🇪🇺",
    layout="wide"
)

st.markdown("""
<style>
.metric-good { color: #28a745; }
.metric-bad { color: #dc3545; }
.fixed-badge { background: #28a745; color: white; padding: 0.2rem; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Ładuje dane europejskie"""
    try:
        data = pd.read_csv('data/raw/european_flights_data.csv')
        data['flight_date'] = pd.to_datetime(data['flight_date'])
        return data
    except:
        return None

def load_models():
    """Ładuje poprawione modele"""
    try:
        classifier = joblib.load('notebooks/european_fixed_model_classifier.joblib')
        regressor = joblib.load('notebooks/european_fixed_model_regressor.joblib')
        return classifier, regressor
    except:
        return None, None

def get_distance_map():
    """Mapa dystansów między lotniskami europejskimi w km"""
    return {
        # Polskie lotniska - główne europejskie
        ('WAW', 'LHR'): 1465, ('WAW', 'CDG'): 1365, ('WAW', 'FRA'): 1160, ('WAW', 'AMS'): 1095,
        ('WAW', 'FCO'): 1315, ('WAW', 'MAD'): 2015, ('WAW', 'BCN'): 1760, ('WAW', 'MUC'): 880,
        ('KRK', 'LHR'): 1555, ('KRK', 'CDG'): 1365, ('KRK', 'FRA'): 950, ('KRK', 'AMS'): 1185,
        ('KRK', 'FCO'): 1105, ('KRK', 'MAD'): 1965, ('KRK', 'BCN'): 1515, ('KRK', 'MUC'): 670,
        ('GDN', 'LHR'): 1465, ('GDN', 'CDG'): 1465, ('GDN', 'FRA'): 1265, ('GDN', 'AMS'): 995,
        ('GDN', 'FCO'): 1565, ('GDN', 'MAD'): 2265, ('GDN', 'BCN'): 2015, ('GDN', 'MUC'): 1135,
        ('WRO', 'LHR'): 1365, ('WRO', 'CDG'): 1165, ('WRO', 'FRA'): 765, ('WRO', 'AMS'): 965,
        ('WRO', 'FCO'): 1215, ('WRO', 'MAD'): 1815, ('WRO', 'BCN'): 1565, ('WRO', 'MUC'): 565,
        ('KTW', 'LHR'): 1465, ('KTW', 'CDG'): 1265, ('KTW', 'FRA'): 865, ('KTW', 'AMS'): 1065,
        ('KTW', 'FCO'): 1115, ('KTW', 'MAD'): 1865, ('KTW', 'BCN'): 1515, ('KTW', 'MUC'): 665,
        ('POZ', 'LHR'): 1265, ('POZ', 'CDG'): 1165, ('POZ', 'FRA'): 765, ('POZ', 'AMS'): 765,
        ('POZ', 'FCO'): 1415, ('POZ', 'MAD'): 2015, ('POZ', 'BCN'): 1765, ('POZ', 'MUC'): 865,
        
        # Między głównymi hubami europejskimi
        ('LHR', 'CDG'): 465, ('LHR', 'FRA'): 815, ('LHR', 'AMS'): 465, ('LHR', 'FCO'): 1465,
        ('LHR', 'MAD'): 1265, ('LHR', 'MUC'): 945, ('CDG', 'FRA'): 465, ('CDG', 'AMS'): 465,
        ('CDG', 'FCO'): 1115, ('CDG', 'MAD'): 1055, ('CDG', 'MUC'): 685, ('FRA', 'AMS'): 365,
        ('FRA', 'FCO'): 965, ('FRA', 'MAD'): 1165, ('FRA', 'MUC'): 305, ('AMS', 'FCO'): 1315,
        ('AMS', 'MAD'): 1465, ('AMS', 'MUC'): 715, ('FCO', 'MAD'): 1365, ('FCO', 'MUC'): 765,
        ('MAD', 'MUC'): 1415,
        
        # Polskie trasy domowe
        ('WAW', 'KRK'): 295, ('WAW', 'GDN'): 345, ('WAW', 'WRO'): 295, ('WAW', 'KTW'): 285,
        ('WAW', 'POZ'): 315, ('KRK', 'GDN'): 565, ('KRK', 'WRO'): 295, ('KRK', 'KTW'): 85,
        ('KRK', 'POZ'): 465, ('GDN', 'WRO'): 315, ('GDN', 'KTW'): 445, ('GDN', 'POZ'): 215,
        ('WRO', 'KTW'): 195, ('WRO', 'POZ'): 165, ('KTW', 'POZ'): 345
    }

def get_distance(origin, destination):
    """Zwraca dystans między lotniskami"""
    distance_map = get_distance_map()
    
    # Sprawdź bezpośrednio
    if (origin, destination) in distance_map:
        return distance_map[(origin, destination)]
    
    # Sprawdź odwrotnie
    if (destination, origin) in distance_map:
        return distance_map[(destination, origin)]
    
    # Domyślna wartość dla nieznanych tras
    return 1000

def main():
    # Header
    st.markdown("# 🇪🇺 European Airline Analytics")
    st.markdown('<span class="fixed-badge">✅ POPRAWIONE MODELE</span>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Menu
    tab1, tab2, tab3 = st.tabs(["📊 Przegląd", "🔮 Przewidywanie", "🧪 Testy"])
    
    # Ładowanie
    data = load_data()
    classifier, regressor = load_models()
    
    # Status w sidebar
    with st.sidebar:
        st.markdown("### 📋 Status")
        if data is not None:
            st.success(f"✅ Dane: {len(data):,} lotów")
        else:
            st.error("❌ Brak danych")
        
        if classifier:
            st.success("✅ Klasyfikator")
        else:
            st.error("❌ Brak klasyfikatora")
        
        if regressor:
            st.success("✅ Regressor")
        else:
            st.error("❌ Brak regressora")
    
    # Tab 1: Przegląd
    with tab1:
        st.header("📊 Przegląd Europejski")
        
        if data is None:
            st.error("❌ Brak danych! Uruchom: `python demo_european_analysis.py`")
            return
        
        # KPIs
        total = len(data)
        delayed = len(data[data['delay_minutes'] > 15])
        on_time_pct = ((total - delayed) / total) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🇪🇺 Łączne loty", f"{total:,}")
        with col2:
            color = "metric-good" if on_time_pct >= 80 else "metric-bad"
            st.markdown(f'<div class="{color}">⏰ Punktualność: {on_time_pct:.1f}%</div>', 
                       unsafe_allow_html=True)
        with col3:
            polish = len(data[(data['country_origin'] == 'Polska') | (data['country_destination'] == 'Polska')])
            st.metric("🇵🇱 Loty polskie", f"{(polish/total)*100:.1f}%")
        
        # Wykres
        daily = data.groupby(data['flight_date'].dt.date)['delay_minutes'].mean().reset_index()
        daily.columns = ['date', 'avg_delay']
        
        fig = px.line(daily, x='date', y='avg_delay', title='📈 Średnie opóźnienia')
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Przewidywanie
    with tab2:
        st.header("🔮 Przewidywanie Opóźnień")
        st.markdown('<span class="fixed-badge">✅ BEZ DATA LEAKAGE</span>', unsafe_allow_html=True)
        
        if not classifier or not regressor:
            st.error("❌ Brak modeli! Uruchom: `python train_european_models_fixed.py`")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            airline = st.selectbox("✈️ Linia lotnicza:", ['LOT Polish Airlines', 'Ryanair', 'Lufthansa', 'Wizz Air', 'easyJet'], key="airline_select")
            origin = st.selectbox("🛫 Lotnisko wylotu:", ['WAW', 'KRK', 'GDN', 'WRO', 'KTW', 'POZ', 'LHR', 'CDG', 'FRA', 'AMS'], key="origin_select")
            destination = st.selectbox("🛬 Lotnisko docelowe:", ['LHR', 'CDG', 'FRA', 'AMS', 'WAW', 'KRK', 'GDN', 'WRO', 'FCO', 'MAD'], key="destination_select")
        
        with col2:
            flight_date = st.date_input("📅 Data lotu:", datetime.now().date(), key="flight_date_input")
            # Alternatywne rozwiązanie - selectbox z predefiniowanymi godzinami
            time_options = [f"{h:02d}:{m:02d}" for h in range(5, 24) for m in [0, 30]]
            departure_time_str = st.selectbox("⏰ Godzina odlotu:", time_options, 
                                            index=time_options.index("08:00"), 
                                            key="departure_time_select", 
                                            help="Wybierz godzinę odlotu")
            departure_time = datetime.strptime(departure_time_str, "%H:%M").time()
            
            # Automatyczne obliczanie dystansu
            auto_distance = get_distance(origin, destination)
            distance = st.number_input("📏 Dystans (km):", value=auto_distance, min_value=100, max_value=3000, step=50, 
                                     help=f"Automatycznie obliczone dla trasy {origin}→{destination}", key="distance_input")
        
        if st.button("🔮 Przewiduj opóźnienie", type="primary", key="predict_button"):
            flight_details = {
                'flight_date': flight_date.strftime('%Y-%m-%d'),
                'airline': airline,
                'origin': origin,
                'destination': destination,
                'country_origin': 'Polska',
                'country_destination': 'Wielka Brytania',
                'distance_km': distance,
                'scheduled_departure': departure_time.strftime('%H:%M'),
                'day_of_week': flight_date.weekday(),
                'month': flight_date.month,
                'hour': departure_time.hour
            }
            
            try:
                prediction = predict_european_delay(flight_details, classifier, regressor)
                
                st.markdown("### 🎯 Wyniki przewidywania:")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    prob = prediction['delay_probability'] * 100
                    st.metric("🎯 Prawdopodobieństwo opóźnienia", f"{prob:.1f}%")
                with col2:
                    status = "OPÓŹNIONY" if prediction['is_delayed'] else "PUNKTUALNY"
                    color = "🔴" if prediction['is_delayed'] else "🟢"
                    st.metric("🚨 Przewidywany status", f"{color} {status}")
                with col3:
                    delay_min = prediction['predicted_delay_minutes']
                    st.metric("⏰ Przewidywane opóźnienie", f"{delay_min:.0f} minut")
                
                # Interpretacja ryzyka po polsku
                risk_pl = {
                    "low": "NISKIE",
                    "medium": "ŚREDNIE", 
                    "high": "WYSOKIE"
                }
                risk_color = {
                    "low": "🟢",
                    "medium": "🟡",
                    "high": "🔴"
                }
                risk_text = risk_pl.get(prediction['delay_risk'], prediction['delay_risk'])
                risk_icon = risk_color.get(prediction['delay_risk'], "⚪")
                
                st.info(f"{risk_icon} **Poziom ryzyka opóźnienia:** {risk_text}")
                
                # Dodatkowe informacje
                st.markdown("#### 📋 Szczegóły lotu:")
                st.write(f"**Trasa:** {origin} → {destination}")
                st.write(f"**Dystans:** {distance} km")
                st.write(f"**Data:** {flight_date.strftime('%d.%m.%Y')} o {departure_time.strftime('%H:%M')}")
                st.write(f"**Linia:** {airline}")
                
            except Exception as e:
                st.error(f"❌ Błąd przewidywania: {str(e)}")
                st.error("Sprawdź czy modele są poprawnie wytrenowane!")
    
    # Tab 3: Testy
    with tab3:
        st.header("🧪 Testy Modeli")
        
        if classifier and hasattr(classifier, 'get_feature_importance'):
            try:
                importance = classifier.get_feature_importance()
                if importance is not None:
                    top10 = importance.head(10)
                    fig = px.bar(top10, x='importance', y='feature', 
                               title='Top 10 Cech', orientation='h')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Sprawdź data leakage
                    suspicious = [f for f in top10['feature'].tolist() 
                                if 'delay' in f.lower() or 'is_delayed' in f.lower()]
                    
                    if suspicious:
                        st.error(f"⚠️ Podejrzane cechy: {suspicious}")
                    else:
                        st.success("✅ Brak data leakage!")
            except Exception as e:
                st.error(f"Błąd: {e}")
        
        if data is not None and st.button("🎲 Test losowy", key="random_test_button"):
            sample = data.sample(1).iloc[0]
            st.write(f"**Test**: {sample['airline']} {sample['origin']}→{sample['destination']}")
            st.write(f"**Rzeczywiste opóźnienie**: {sample['delay_minutes']:.0f} min")

if __name__ == "__main__":
    main() 