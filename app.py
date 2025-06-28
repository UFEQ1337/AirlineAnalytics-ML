#!/usr/bin/env python3
"""
🚀 AIRLINE ANALYTICS ML - MAIN APPLICATION ENTRY POINT
======================================================

Główny plik aplikacji - entry point dla dashboard Streamlit
Obsługuje routing, konfigurację i inicjalizację systemu

Autor: AirlineAnalytics-ML Team
"""

import streamlit as st
import sys
import os
import logging
from datetime import datetime
import warnings

# Konfiguracja warnings
warnings.filterwarnings('ignore')

# Dodaj src do ścieżki
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ===========================
# INICJALIZACJA I HEALTH CHECK
# ===========================

def health_check():
    """Sprawdza czy wszystkie wymagane komponenty są dostępne"""
    issues = []
    
    # Sprawdź katalogi
    required_dirs = ['src', 'notebooks', 'data/processed', 'data/raw']
    for directory in required_dirs:
        dir_path = os.path.join(current_dir, directory)
        if not os.path.exists(dir_path):
            issues.append(f"❌ Brak katalogu: {directory}")
    
    # Sprawdź kluczowe pliki
    required_files = [
        'data/processed/flights_cleaned.csv',
        'notebooks/best_model_classifier.joblib',
        'notebooks/best_model_regressor.joblib',
        'src/dashboard.py'
    ]
    
    for file_path in required_files:
        full_path = os.path.join(current_dir, file_path)
        if not os.path.exists(full_path):
            issues.append(f"❌ Brak pliku: {file_path}")
    
    return issues

def main():
    """Główna funkcja aplikacji"""
    # Sprawdź health system
    issues = health_check()
    
    if issues:
        st.error("🚨 **Problemy z konfiguracją systemu:**")
        for issue in issues:
            st.write(issue)
        st.stop()
    
    # Wszystko OK - uruchom dashboard
    try:
        # Import i uruchom dashboard
        import dashboard
        # Dashboard ma własną logikę w src/dashboard.py
        
    except ImportError as e:
        st.error(f"❌ Błąd importu dashboard: {str(e)}")
        st.info("Spróbuj uruchomić bezpośrednio: `streamlit run src/dashboard.py`")
    except Exception as e:
        st.error(f"❌ Nieoczekiwany błąd: {str(e)}")

if __name__ == "__main__":
    main()