"""
🔧 UTILITIES - FUNKCJE POMOCNICZE
==============================

Moduł zawiera funkcje pomocnicze dla projektu AirlineAnalytics-ML:
- Ładowanie i czyszczenie danych
- Generowanie raportów
- Eksport do PDF
- Monitoring zdrowia modelu
- Inne funkcje pomocnicze

Autor: AirlineAnalytics-ML Team
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ===========================
# ŁADOWANIE DANYCH
# ===========================

def load_clean_data(data_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    🔄 Ładuje oczyszczone dane lotów
    
    Args:
        data_path (str, optional): Ścieżka do pliku z danymi
        
    Returns:
        pd.DataFrame: Oczyszczone dane lotów lub None w przypadku błędu
    """
    if data_path is None:
        # Domyślna ścieżka
        base_path = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(base_path, 'data', 'processed', 'flights_cleaned.csv')
    
    try:
        print(f"🔄 Ładowanie danych z: {data_path}")
        
        if not os.path.exists(data_path):
            print(f"❌ Plik nie istnieje: {data_path}")
            return None
        
        # Wczytaj dane
        df = pd.read_csv(data_path)
        
        # Podstawowe czyszczenie i konwersja typów
        df['flight_date'] = pd.to_datetime(df['flight_date'])
        
        # Sprawdź czy mamy wymagane kolumny
        required_columns = ['flight_date', 'airline', 'origin', 'destination', 'delay_minutes']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ Brakujące kolumny: {missing_columns}")
            return None
        
        # Podstawowe statystyki
        print(f"✅ Dane załadowane pomyślnie!")
        print(f"   📊 Liczba rekordów: {len(df):,}")
        print(f"   📅 Zakres dat: {df['flight_date'].min().date()} - {df['flight_date'].max().date()}")
        print(f"   ✈️ Liczba linii: {df['airline'].nunique()}")
        print(f"   🛫 Liczba lotnisk: {df['origin'].nunique()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Błąd ładowania danych: {str(e)}")
        return None

def load_raw_data(data_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    🔄 Ładuje surowe dane lotów
    
    Args:
        data_path (str, optional): Ścieżka do pliku z danymi
        
    Returns:
        pd.DataFrame: Surowe dane lotów lub None w przypadku błędu
    """
    if data_path is None:
        base_path = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(base_path, 'data', 'raw', 'flights_data.csv')
    
    try:
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['flight_date'] = pd.to_datetime(df['flight_date'])
            return df
        else:
            print(f"❌ Plik nie istnieje: {data_path}")
            return None
    except Exception as e:
        print(f"❌ Błąd ładowania surowych danych: {str(e)}")
        return None

# ===========================
# GENEROWANIE RAPORTÓW
# ===========================

def generate_report(start_date: str, end_date: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    📊 Generuje raport analityczny dla określonego zakresu dat
    
    Args:
        start_date (str): Data początkowa (YYYY-MM-DD)
        end_date (str): Data końcowa (YYYY-MM-DD)
        output_path (str, optional): Ścieżka do zapisu raportu
        
    Returns:
        Dict[str, Any]: Słownik z wynikami analizy
    """
    print(f"📊 Generowanie raportu za okres: {start_date} - {end_date}")
    
    try:
        # Załaduj dane
        df = load_clean_data()
        if df is None:
            return {"error": "Nie można załadować danych"}
        
        # Filtruj dane według zakresu dat
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        filtered_df = df[
            (df['flight_date'] >= start_dt) & 
            (df['flight_date'] <= end_dt)
        ]
        
        if filtered_df.empty:
            return {"error": "Brak danych dla wybranego okresu"}
        
        # === KLUCZOWE METRYKI ===
        total_flights = len(filtered_df)
        avg_delay = filtered_df['delay_minutes'].mean()
        median_delay = filtered_df['delay_minutes'].median()
        on_time_flights = (filtered_df['delay_minutes'] <= 15).sum()
        on_time_percentage = (on_time_flights / total_flights) * 100
        
        # Opóźnienia znaczące (>60 min)
        significant_delays = (filtered_df['delay_minutes'] > 60).sum()
        significant_delay_percentage = (significant_delays / total_flights) * 100
        
        # === ANALIZA LINII LOTNICZYCH ===
        airline_stats = filtered_df.groupby('airline').agg({
            'delay_minutes': ['mean', 'count', 'std'],
            'flight_date': 'first'
        }).reset_index()
        airline_stats.columns = ['airline', 'avg_delay', 'flight_count', 'delay_std', 'first_flight']
        airline_stats['on_time_percent'] = filtered_df.groupby('airline').apply(
            lambda x: (x['delay_minutes'] <= 15).sum() / len(x) * 100
        ).values
        
        # Top 5 najlepszych i najgorszych linii
        airline_stats_filtered = airline_stats[airline_stats['flight_count'] >= 10]
        best_airlines = airline_stats_filtered.nsmallest(5, 'avg_delay')
        worst_airlines = airline_stats_filtered.nlargest(5, 'avg_delay')
        
        # === ANALIZA LOTNISK ===
        airport_stats = filtered_df.groupby('origin').agg({
            'delay_minutes': ['mean', 'count'],
            'flight_date': 'first'
        }).reset_index()
        airport_stats.columns = ['airport', 'avg_delay', 'flight_count', 'first_flight']
        airport_stats_filtered = airport_stats[airport_stats['flight_count'] >= 10]
        
        best_airports = airport_stats_filtered.nsmallest(5, 'avg_delay')
        worst_airports = airport_stats_filtered.nlargest(5, 'avg_delay')
        
        # === ANALIZA CZASOWA ===
        # Analiza według dnia tygodnia
        filtered_df['day_of_week'] = filtered_df['flight_date'].dt.dayofweek
        filtered_df['day_name'] = filtered_df['flight_date'].dt.day_name()
        
        daily_stats = filtered_df.groupby('day_name').agg({
            'delay_minutes': ['mean', 'count'],
            'flight_date': 'first'
        }).reset_index()
        daily_stats.columns = ['day', 'avg_delay', 'flight_count', 'first_flight']
        
        # Analiza według godziny
        filtered_df['hour'] = pd.to_datetime(filtered_df['scheduled_departure']).dt.hour
        hourly_stats = filtered_df.groupby('hour').agg({
            'delay_minutes': ['mean', 'count'],
            'flight_date': 'first'
        }).reset_index()
        hourly_stats.columns = ['hour', 'avg_delay', 'flight_count', 'first_flight']
        
        # === TRENDY ===
        # Trend tygodniowy
        weekly_trend = filtered_df.groupby(filtered_df['flight_date'].dt.date).agg({
            'delay_minutes': 'mean',
            'flight_date': 'first'
        }).reset_index()
        weekly_trend.columns = ['date', 'avg_delay', 'flight_date_sample']
        
        # Oblicz trend (prosta regresja)
        if len(weekly_trend) > 1:
            x = np.arange(len(weekly_trend))
            y = weekly_trend['avg_delay'].values
            trend_slope = np.polyfit(x, y, 1)[0]
            trend_direction = "rosnący" if trend_slope > 0.1 else "malejący" if trend_slope < -0.1 else "stabilny"
        else:
            trend_direction = "brak danych"
            trend_slope = 0
        
        # === PRZYGOTOWANIE RAPORTU ===
        report = {
            "period": {
                "start_date": start_date,
                "end_date": end_date,
                "days": (end_dt - start_dt).days + 1
            },
            "summary": {
                "total_flights": int(total_flights),
                "avg_delay_minutes": round(avg_delay, 2),
                "median_delay_minutes": round(median_delay, 2),
                "on_time_flights": int(on_time_flights),
                "on_time_percentage": round(on_time_percentage, 2),
                "significant_delays": int(significant_delays),
                "significant_delay_percentage": round(significant_delay_percentage, 2)
            },
            "airlines": {
                "best_airlines": best_airlines[['airline', 'avg_delay', 'on_time_percent']].to_dict('records'),
                "worst_airlines": worst_airlines[['airline', 'avg_delay', 'on_time_percent']].to_dict('records'),
                "total_airlines": int(filtered_df['airline'].nunique())
            },
            "airports": {
                "best_airports": best_airports[['airport', 'avg_delay', 'flight_count']].to_dict('records'),
                "worst_airports": worst_airports[['airport', 'avg_delay', 'flight_count']].to_dict('records'),
                "total_airports": int(filtered_df['origin'].nunique())
            },
            "temporal_analysis": {
                "daily_stats": daily_stats[['day', 'avg_delay', 'flight_count']].to_dict('records'),
                "peak_hours": hourly_stats.nlargest(3, 'avg_delay')[['hour', 'avg_delay']].to_dict('records'),
                "trend": {
                    "direction": trend_direction,
                    "slope": round(trend_slope, 4)
                }
            },
            "generated_at": datetime.now().isoformat(),
            "data_quality": {
                "total_records": int(len(df)),
                "filtered_records": int(len(filtered_df)),
                "coverage_percentage": round((len(filtered_df) / len(df)) * 100, 2)
            }
        }
        
        # Zapisz raport do pliku JSON
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"✅ Raport zapisany do: {output_path}")
        
        print(f"✅ Raport wygenerowany pomyślnie!")
        print(f"   📊 Przeanalizowano {total_flights:,} lotów")
        print(f"   ⏱️ Średnie opóźnienie: {avg_delay:.1f} minut")
        print(f"   ✅ Punktualność: {on_time_percentage:.1f}%")
        
        return report
        
    except Exception as e:
        print(f"❌ Błąd generowania raportu: {str(e)}")
        return {"error": str(e)}

# ===========================
# EKSPORT DO PDF
# ===========================

def export_insights_to_pdf(report_data: Dict[str, Any], output_path: str) -> bool:
    """
    📄 Eksportuje insights do pliku PDF
    
    Args:
        report_data (Dict): Dane raportu z generate_report()
        output_path (str): Ścieżka do zapisu PDF
        
    Returns:
        bool: True jeśli eksport się udał
    """
    try:
        print(f"📄 Eksportowanie insights do PDF: {output_path}")
        
        # Tworzenie dokumentu PDF
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=HexColor('#1f4e79'),
            alignment=1  # Center alignment
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=HexColor('#2c3e50'),
            leftIndent=0
        )
        
        # Tytuł
        story.append(Paragraph("✈️ AIRLINE ANALYTICS ML - RAPORT INSIGHTS", title_style))
        story.append(Spacer(1, 20))
        
        # Okres analizy
        period = report_data.get('period', {})
        story.append(Paragraph(f"📅 Okres analizy: {period.get('start_date')} - {period.get('end_date')}", styles['Normal']))
        story.append(Paragraph(f"🗓️ Liczba dni: {period.get('days', 0)}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Podsumowanie wykonawcze
        story.append(Paragraph("📊 PODSUMOWANIE WYKONAWCZE", heading_style))
        
        summary = report_data.get('summary', {})
        summary_data = [
            ['Metryka', 'Wartość'],
            ['Łączna liczba lotów', f"{summary.get('total_flights', 0):,}"],
            ['Średnie opóźnienie', f"{summary.get('avg_delay_minutes', 0):.1f} minut"],
            ['Punktualność (≤15 min)', f"{summary.get('on_time_percentage', 0):.1f}%"],
            ['Opóźnienia znaczące (>60 min)', f"{summary.get('significant_delay_percentage', 0):.1f}%"]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Top 3 insights
        story.append(Paragraph("🔍 KLUCZOWE ODKRYCIA", heading_style))
        
        # Najlepsza linia
        airlines = report_data.get('airlines', {})
        best_airlines = airlines.get('best_airlines', [])
        if best_airlines:
            best_airline = best_airlines[0]
            story.append(Paragraph(f"🏆 <b>Najlepsza linia lotnicza:</b> {best_airline.get('airline', 'N/A')} - {best_airline.get('avg_delay', 0):.1f} min średniego opóźnienia", styles['Normal']))
        
        # Najgorsze lotnisko
        airports = report_data.get('airports', {})
        worst_airports = airports.get('worst_airports', [])
        if worst_airports:
            worst_airport = worst_airports[0]
            story.append(Paragraph(f"🔴 <b>Najgorsze lotnisko:</b> {worst_airport.get('airport', 'N/A')} - {worst_airport.get('avg_delay', 0):.1f} min średniego opóźnienia", styles['Normal']))
        
        # Trend
        temporal = report_data.get('temporal_analysis', {})
        trend = temporal.get('trend', {})
        story.append(Paragraph(f"📈 <b>Trend opóźnień:</b> {trend.get('direction', 'nieznany')}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Analiza linii lotniczych
        story.append(Paragraph("✈️ ANALIZA LINII LOTNICZYCH", heading_style))
        
        # Top 5 najlepszych linii
        if best_airlines:
            story.append(Paragraph("<b>Top 5 najlepszych linii:</b>", styles['Normal']))
            airlines_data = [['Pozycja', 'Linia', 'Śr. opóźnienie (min)', 'Punktualność (%)']]
            for i, airline in enumerate(best_airlines[:5], 1):
                airlines_data.append([
                    str(i),
                    airline.get('airline', 'N/A'),
                    f"{airline.get('avg_delay', 0):.1f}",
                    f"{airline.get('on_time_percent', 0):.1f}"
                ])
            
            airlines_table = Table(airlines_data, colWidths=[0.7*inch, 2*inch, 1.3*inch, 1.3*inch])
            airlines_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.green),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(airlines_table)
        
        story.append(Spacer(1, 15))
        
        # Rekomendacje biznesowe
        story.append(Paragraph("💡 REKOMENDACJE BIZNESOWE", heading_style))
        
        recommendations = [
            "1. Skupić się na poprawie punktualności linii o najgorszych wynikach",
            "2. Analizować wzorce opóźnień w godzinach szczytu",
            "3. Wdrożyć system wczesnego ostrzegania o potencjalnych opóźnieniach",
            "4. Negocjować lepsze sloty czasowe na lotniskach problematycznych",
            "5. Monitorować trendy opóźnień w czasie rzeczywistym"
        ]
        
        for rec in recommendations:
            story.append(Paragraph(rec, styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Footer
        story.append(Paragraph(f"📋 Raport wygenerowany: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph("🔬 AirlineAnalytics-ML Team", styles['Normal']))
        
        # Budowanie PDF
        doc.build(story)
        
        print(f"✅ Raport PDF zapisany pomyślnie: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Błąd eksportu PDF: {str(e)}")
        return False

# ===========================
# MONITORING ZDROWIA MODELU
# ===========================

def model_health_check() -> Dict[str, Any]:
    """
    🏥 Sprawdza zdrowie modeli ML
    
    Returns:
        Dict[str, Any]: Status zdrowia modeli
    """
    try:
        print("🏥 Sprawdzanie zdrowia modeli...")
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "checks": {}
        }
        
        # Sprawdzenie istnienia plików modeli
        base_path = os.path.dirname(os.path.dirname(__file__))
        classifier_path = os.path.join(base_path, 'notebooks', 'best_model_classifier.joblib')
        regressor_path = os.path.join(base_path, 'notebooks', 'best_model_regressor.joblib')
        
        # Test 1: Istnienie plików
        classifier_exists = os.path.exists(classifier_path)
        regressor_exists = os.path.exists(regressor_path)
        
        health_status["checks"]["classifier_file"] = {
            "status": "pass" if classifier_exists else "fail",
            "message": "Plik klasyfikatora istnieje" if classifier_exists else "Brak pliku klasyfikatora"
        }
        
        health_status["checks"]["regressor_file"] = {
            "status": "pass" if regressor_exists else "fail", 
            "message": "Plik regresora istnieje" if regressor_exists else "Brak pliku regresora"
        }
        
        # Test 2: Ładowanie modeli
        try:
            if classifier_exists:
                classifier = joblib.load(classifier_path)
                health_status["checks"]["classifier_load"] = {
                    "status": "pass",
                    "message": "Klasyfikator załadowany pomyślnie"
                }
            else:
                health_status["checks"]["classifier_load"] = {
                    "status": "fail",
                    "message": "Nie można załadować klasyfikatora"
                }
                
            if regressor_exists:
                regressor = joblib.load(regressor_path)
                health_status["checks"]["regressor_load"] = {
                    "status": "pass",
                    "message": "Regresor załadowany pomyślnie"
                }
            else:
                health_status["checks"]["regressor_load"] = {
                    "status": "fail",
                    "message": "Nie można załadować regresora"
                }
                
        except Exception as e:
            health_status["checks"]["model_load"] = {
                "status": "fail",
                "message": f"Błąd ładowania modeli: {str(e)}"
            }
        
        # Test 3: Sprawdzenie danych
        data = load_clean_data()
        if data is not None:
            health_status["checks"]["data_availability"] = {
                "status": "pass",
                "message": f"Dane dostępne: {len(data):,} rekordów"
            }
            
            # Sprawdź czy dane są aktualne (ostatnie 60 dni)
            latest_date = data['flight_date'].max()
            days_since_latest = (datetime.now() - latest_date).days
            
            if days_since_latest <= 7:
                health_status["checks"]["data_freshness"] = {
                    "status": "pass",
                    "message": f"Dane bardzo aktualne (ostatnie: {latest_date.date()})"
                }
            elif days_since_latest <= 60:
                health_status["checks"]["data_freshness"] = {
                    "status": "pass",
                    "message": f"Dane aktualne (ostatnie: {latest_date.date()}, {days_since_latest} dni temu)"
                }
            else:
                health_status["checks"]["data_freshness"] = {
                    "status": "warning",
                    "message": f"Dane nieaktualne (ostatnie: {latest_date.date()}, {days_since_latest} dni temu)"
                }
        else:
            health_status["checks"]["data_availability"] = {
                "status": "fail",
                "message": "Brak dostępu do danych"
            }
        
        # Test 4: Test predykcji (jeśli modele są dostępne)
        if classifier_exists and regressor_exists and data is not None:
            try:
                from models import predict_delay
                
                # Testowe dane lotu
                test_flight = {
                    'airline': 'American Airlines',
                    'origin': 'JFK',
                    'destination': 'LAX',
                    'aircraft_type': 'B737',
                    'distance_miles': 2500,
                    'scheduled_departure': '10:00',
                    'flight_date': datetime.now().strftime('%Y-%m-%d')
                }
                
                classifier = joblib.load(classifier_path)
                regressor = joblib.load(regressor_path)
                
                prediction = predict_delay(test_flight, classifier, regressor)
                
                if prediction:
                    health_status["checks"]["prediction_test"] = {
                        "status": "pass",
                        "message": "Test predykcji zakończony pomyślnie"
                    }
                else:
                    health_status["checks"]["prediction_test"] = {
                        "status": "fail",
                        "message": "Test predykcji nieudany"
                    }
                    
            except Exception as e:
                health_status["checks"]["prediction_test"] = {
                    "status": "fail",
                    "message": f"Błąd testu predykcji: {str(e)}"
                }
        
        # Określ ogólny status
        failed_checks = sum(1 for check in health_status["checks"].values() if check["status"] == "fail")
        warning_checks = sum(1 for check in health_status["checks"].values() if check["status"] == "warning")
        
        if failed_checks == 0 and warning_checks == 0:
            health_status["overall_status"] = "healthy"
        elif failed_checks == 0:
            health_status["overall_status"] = "warning"
        else:
            health_status["overall_status"] = "unhealthy"
        
        print(f"✅ Sprawdzenie zdrowia zakończone. Status: {health_status['overall_status']}")
        return health_status
        
    except Exception as e:
        print(f"❌ Błąd sprawdzania zdrowia modelu: {str(e)}")
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "error",
            "error": str(e)
        }

# ===========================
# DODATKOWE FUNKCJE POMOCNICZE
# ===========================

def create_backup(source_path: str, backup_dir: str = "backups") -> bool:
    """
    💾 Tworzy kopię zapasową pliku
    
    Args:
        source_path (str): Ścieżka do pliku źródłowego
        backup_dir (str): Katalog na backupy
        
    Returns:
        bool: True jeśli backup się udał
    """
    try:
        if not os.path.exists(source_path):
            print(f"❌ Plik źródłowy nie istnieje: {source_path}")
            return False
        
        # Stwórz katalog backup jeśli nie istnieje
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        # Nazwa pliku backup z timestampem
        filename = os.path.basename(source_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{timestamp}_{filename}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Kopiuj plik
        import shutil
        shutil.copy2(source_path, backup_path)
        
        print(f"✅ Backup utworzony: {backup_path}")
        return True
        
    except Exception as e:
        print(f"❌ Błąd tworzenia backup: {str(e)}")
        return False

def get_data_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    📊 Zwraca podstawowe statystyki danych
    
    Args:
        df (pd.DataFrame): DataFrame z danymi
        
    Returns:
        Dict[str, Any]: Słownik ze statystykami
    """
    if df is None or df.empty:
        return {"error": "Brak danych"}
    
    try:
        stats = {
            "basic_info": {
                "total_records": len(df),
                "total_columns": len(df.columns),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            },
            "date_range": {
                "start_date": df['flight_date'].min().date().isoformat() if 'flight_date' in df.columns else None,
                "end_date": df['flight_date'].max().date().isoformat() if 'flight_date' in df.columns else None,
                "date_span_days": (df['flight_date'].max() - df['flight_date'].min()).days if 'flight_date' in df.columns else None
            },
            "delay_statistics": {
                "avg_delay": round(df['delay_minutes'].mean(), 2) if 'delay_minutes' in df.columns else None,
                "median_delay": round(df['delay_minutes'].median(), 2) if 'delay_minutes' in df.columns else None,
                "max_delay": round(df['delay_minutes'].max(), 2) if 'delay_minutes' in df.columns else None,
                "min_delay": round(df['delay_minutes'].min(), 2) if 'delay_minutes' in df.columns else None,
                "std_delay": round(df['delay_minutes'].std(), 2) if 'delay_minutes' in df.columns else None
            },
            "categorical_counts": {
                "airlines": int(df['airline'].nunique()) if 'airline' in df.columns else None,
                "airports": int(df['origin'].nunique()) if 'origin' in df.columns else None,
                "destinations": int(df['destination'].nunique()) if 'destination' in df.columns else None
            },
            "data_quality": {
                "missing_values": df.isnull().sum().to_dict(),
                "duplicate_rows": int(df.duplicated().sum())
            }
        }
        
        return stats
        
    except Exception as e:
        return {"error": f"Błąd obliczania statystyk: {str(e)}"}

def validate_data_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    🔍 Sprawdza integralność danych
    
    Args:
        df (pd.DataFrame): DataFrame do sprawdzenia
        
    Returns:
        Dict[str, Any]: Raport z integralności
    """
    if df is None or df.empty:
        return {"error": "Brak danych do sprawdzenia"}
    
    try:
        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "issues": [],
            "warnings": [],
            "status": "unknown"
        }
        
        # Sprawdź braki danych
        missing_data = df.isnull().sum()
        critical_columns = ['flight_date', 'airline', 'origin', 'destination', 'delay_minutes']
        
        for col in critical_columns:
            if col in df.columns:
                missing_count = missing_data[col]
                if missing_count > 0:
                    missing_percent = (missing_count / len(df)) * 100
                    if missing_percent > 10:
                        validation_report["issues"].append(f"Kolumna '{col}': {missing_count} braków ({missing_percent:.1f}%)")
                    elif missing_percent > 5:
                        validation_report["warnings"].append(f"Kolumna '{col}': {missing_count} braków ({missing_percent:.1f}%)")
            else:
                validation_report["issues"].append(f"Brak krytycznej kolumny: '{col}'")
        
        # Sprawdź zakresy dat
        if 'flight_date' in df.columns:
            date_range = df['flight_date'].max() - df['flight_date'].min()
            if date_range.days > 365:
                validation_report["warnings"].append(f"Szeroki zakres dat: {date_range.days} dni")
        
        # Sprawdź wartości opóźnień
        if 'delay_minutes' in df.columns:
            extreme_delays = (df['delay_minutes'] > 300).sum()  # >5 godzin
            if extreme_delays > 0:
                validation_report["warnings"].append(f"Ekstremalne opóźnienia (>5h): {extreme_delays}")
            
            negative_delays = (df['delay_minutes'] < -60).sum()  # <-1 godzina
            if negative_delays > 0:
                validation_report["warnings"].append(f"Podejrzane negatywne opóźnienia (<-1h): {negative_delays}")
        
        # Sprawdź duplikaty
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            dup_percent = (duplicates / len(df)) * 100
            if dup_percent > 5:
                validation_report["issues"].append(f"Duplikaty: {duplicates} ({dup_percent:.1f}%)")
            else:
                validation_report["warnings"].append(f"Duplikaty: {duplicates} ({dup_percent:.1f}%)")
        
        # Określ status
        if len(validation_report["issues"]) == 0:
            validation_report["status"] = "good" if len(validation_report["warnings"]) == 0 else "warning"
        else:
            validation_report["status"] = "issues"
        
        return validation_report
        
    except Exception as e:
        return {"error": f"Błąd walidacji danych: {str(e)}"}

if __name__ == "__main__":
    print("🔧 AIRLINE ANALYTICS ML - UTILITIES")
    print("=" * 50)
    
    # Test podstawowych funkcji
    print("\n1. 🔄 Test ładowania danych...")
    data = load_clean_data()
    
    if data is not None:
        print("\n2. 📊 Statystyki danych...")
        stats = get_data_statistics(data)
        print(f"   Rekordy: {stats['basic_info']['total_records']:,}")
        print(f"   Kolumny: {stats['basic_info']['total_columns']}")
        
        print("\n3. 🔍 Sprawdzenie integralności...")
        integrity = validate_data_integrity(data)
        print(f"   Status: {integrity['status']}")
        
        print("\n4. 🏥 Sprawdzenie zdrowia modeli...")
        health = model_health_check()
        print(f"   Status: {health['overall_status']}")
        
        print("\n5. 📋 Test generowania raportu...")
        start_date = data['flight_date'].min().strftime('%Y-%m-%d')
        end_date = data['flight_date'].max().strftime('%Y-%m-%d')
        report = generate_report(start_date, end_date)
        
        if 'error' not in report:
            print(f"   ✅ Raport wygenerowany dla {report['summary']['total_flights']:,} lotów")
        else:
            print(f"   ❌ Błąd raportu: {report['error']}")
    
    print("\n✅ Test utilities zakończony!") 