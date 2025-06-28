#!/usr/bin/env python3
"""
🇪🇺 TRENOWANIE EUROPEJSKICH MODELI ML
===================================

Skrypt do szybkiego trenowania modeli ML na europejskich danych.
Uruchamia pełny pipeline: dane → feature engineering → trenowanie → ewaluacja → zapis.

Użycie:
    python train_european_models.py

Wymagania:
    - Wygenerowane dane europejskie (data/raw/european_flights_data.csv)
    - Zainstalowane biblioteki ML (sklearn, xgboost)
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Dodaj src do ścieżki
sys.path.append('src')

from european_models import EuropeanDelayClassifier, EuropeanDelayRegressor
from european_models import save_european_models, predict_european_delay

def main():
    """Główna funkcja trenowania europejskich modeli"""
    
    print("🇪🇺 TRENOWANIE EUROPEJSKICH MODELI ML")
    print("="*50)
    print(f"🕐 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. ŁADOWANIE DANYCH
    print("\n📂 KROK 1: Ładowanie danych europejskich...")
    try:
        df_eu = pd.read_csv('data/raw/european_flights_data.csv')
        print(f"✅ Załadowano {len(df_eu):,} europejskich lotów")
        print(f"📊 Opóźnienia >15 min: {(df_eu['delay_minutes'] > 15).mean()*100:.1f}%")
        
        # Sprawdź polskie loty
        polish_flights = df_eu[(df_eu['country_origin'] == 'Polska') | 
                              (df_eu['country_destination'] == 'Polska')]
        print(f"🇵🇱 Polskie loty: {len(polish_flights):,} ({len(polish_flights)/len(df_eu)*100:.1f}%)")
        
    except FileNotFoundError:
        print("❌ Brak pliku z danymi europejskimi!")
        print("💡 Najpierw uruchom: python src/data_generator_eu.py")
        return
    except Exception as e:
        print(f"❌ Błąd ładowania danych: {e}")
        return
    
    # 2. TRENOWANIE KLASYFIKATORA
    print("\n🎯 KROK 2: Trenowanie klasyfikatora europejskiego...")
    try:
        classifier = EuropeanDelayClassifier(model_type='xgboost')
        
        # Feature engineering
        X_clf, y_clf = classifier.prepare_european_features(df_eu)
        print(f"📊 Features: {X_clf.shape[1]}, Samples: {X_clf.shape[0]:,}")
        
        # Trenowanie
        if y_clf is not None:
            results_clf = classifier.train(X_clf, y_clf, test_size=0.2, hyperparameter_tuning=True)
            print(f"✅ Klasyfikator wytrenowany - ROC AUC: {results_clf['roc_auc']:.3f}")
        else:
            print("❌ Błąd: brak target variable")
            return
            
    except Exception as e:
        print(f"❌ Błąd trenowania klasyfikatora: {e}")
        return
    
    # 3. TRENOWANIE REGRESSORA
    print("\n📈 KROK 3: Trenowanie regressora europejskiego...")
    try:
        regressor = EuropeanDelayRegressor(model_type='xgboost')
        
        # Feature engineering (tylko opóźnione loty)
        X_reg, y_reg = regressor.prepare_features(df_eu)
        print(f"📊 Opóźnione loty: {X_reg.shape[0]:,}, Średnie opóźnienie: {y_reg.mean():.1f} min")
        
        # Trenowanie
        results_reg = regressor.train(X_reg, y_reg, test_size=0.2)
        print(f"✅ Regressor wytrenowany - R²: {results_reg['test_r2']:.3f}, MAE: {results_reg['mae']:.1f} min")
        
    except Exception as e:
        print(f"❌ Błąd trenowania regressora: {e}")
        return
    
    # 4. ZAPISYWANIE MODELI
    print("\n💾 KROK 4: Zapisywanie modeli...")
    try:
        classifier_path, regressor_path = save_european_models(classifier, regressor)
        print(f"✅ Modele zapisane pomyślnie!")
        
    except Exception as e:
        print(f"❌ Błąd zapisywania: {e}")
        return
    
    # 5. TESTOWANIE PRZEWIDYWAŃ
    print("\n🔮 KROK 5: Testowanie przewidywań...")
    test_flights = [
        {
            'flight_date': '2024-07-15',
            'airline': 'LOT Polish Airlines',
            'origin': 'WAW',
            'destination': 'LHR',
            'country_origin': 'Polska',
            'country_destination': 'Wielka Brytania',
            'scheduled_departure': '08:30',
            'distance_km': 1460,
            'aircraft_type': 'Boeing 737-800',
            'delay_reason': 'None'
        },
        {
            'flight_date': '2024-12-20',
            'airline': 'Ryanair',
            'origin': 'KRK',
            'destination': 'FRA',
            'country_origin': 'Polska',
            'country_destination': 'Niemcy',
            'scheduled_departure': '06:45',
            'distance_km': 430,
            'aircraft_type': 'Boeing 737-800',
            'delay_reason': 'None'
        },
        {
            'flight_date': '2024-01-15',
            'airline': 'Lufthansa',
            'origin': 'FRA',
            'destination': 'WAW',
            'country_origin': 'Niemcy',
            'country_destination': 'Polska',
            'scheduled_departure': '18:30',
            'distance_km': 1160,
            'aircraft_type': 'Airbus A320',
            'delay_reason': 'None'
        }
    ]
    
    for i, flight in enumerate(test_flights, 1):
        print(f"\n🛫 Test {i}: {flight['airline']} {flight['origin']}→{flight['destination']}")
        print(f"   📅 {flight['flight_date']} o {flight['scheduled_departure']}")
        
        try:
            prediction = predict_european_delay(flight, classifier, regressor)
            
            print(f"   📊 Prawdopodobieństwo opóźnienia: {prediction['delay_probability']:.1%}")
            print(f"   ⏰ Przewidywane opóźnienie: {prediction['predicted_delay_minutes']:.0f} min")
            print(f"   🎯 Ryzyko: {prediction['delay_risk']}")
            print(f"   📝 Kategoria: {prediction['delay_category']}")
            
        except Exception as e:
            print(f"   ❌ Błąd przewidywania: {e}")
    
    # 6. ANALIZA WAŻNOŚCI CECH
    print("\n🏆 KROK 6: Najważniejsze cechy europejskie...")
    try:
        importance_df = classifier.get_feature_importance()
        if importance_df is not None:
            print("Top 10 najważniejszych cech:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                feature_type = get_feature_type(row['feature'])
                print(f"   {i:2d}. {feature_type} {row['feature']}: {row['importance']:.4f}")
        
        # Pokaż cechy polskie
        polish_features = [f for f in classifier.feature_names 
                          if 'polish' in f.lower() or 'poland' in f.lower()]
        print(f"\n🇵🇱 Cechy polskie ({len(polish_features)}):")
        for feature in polish_features[:5]:  # Pokaż pierwsze 5
            print(f"   • {feature}")
            
    except Exception as e:
        print(f"❌ Błąd analizy ważności: {e}")
    
    # 7. PODSUMOWANIE
    print(f"\n✅ TRENOWANIE ZAKOŃCZONE POMYŚLNIE!")
    print("="*50)
    print(f"🎯 Klasyfikator:")
    print(f"   • Dokładność: {results_clf['test_accuracy']:.3f}")
    print(f"   • ROC AUC: {results_clf['roc_auc']:.3f}")
    print(f"   • F1 Score: {results_clf['f1_score']:.3f}")
    
    print(f"\n📈 Regressor:")
    print(f"   • R² Score: {results_reg['test_r2']:.3f}")
    print(f"   • MAE: {results_reg['mae']:.1f} min")
    print(f"   • RMSE: {results_reg['rmse']:.1f} min")
    
    print(f"\n💾 Pliki modeli:")
    print(f"   • {classifier_path}")
    print(f"   • {regressor_path}")
    
    print(f"\n🕐 Zakończono: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📝 NASTĘPNE KROKI:")
    print("1. Uruchom notebook: notebooks/06_european_ml_models.ipynb")
    print("2. Sprawdź wykresy ważności cech")
    print("3. Przetestuj przewidywania na własnych danych")
    print("4. Dostosuj hyperparametry w razie potrzeby")
    print("5. Zintegruj modele z aplikacją dashboard")

def get_feature_type(feature_name):
    """Zwraca emoji dla typu cechy"""
    if 'polish' in feature_name.lower() or 'poland' in feature_name.lower():
        return '🇵🇱'
    elif any(x in feature_name.lower() for x in ['airline', 'lot', 'carrier']):
        return '✈️'
    elif any(x in feature_name.lower() for x in ['region', 'country']):
        return '🌍'
    elif any(x in feature_name.lower() for x in ['hour', 'day', 'month', 'time']):
        return '⏰'
    elif 'distance' in feature_name.lower():
        return '📏'
    else:
        return '🔧'

if __name__ == "__main__":
    print("🚀 Uruchamianie trenowania europejskich modeli...")
    
    try:
        main()
        print(f"\n🎉 SUKCES! Europejskie modele ML gotowe do użycia.")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Przerwano przez użytkownika")
        
    except Exception as e:
        print(f"\n💥 BŁĄD KRYTYCZNY: {e}")
        print("📧 Sprawdź logi i spróbuj ponownie")
        
    finally:
        print(f"\n👋 Koniec pracy skryptu") 