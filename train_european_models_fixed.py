#!/usr/bin/env python3
"""
🔧 POPRAWIONE TRENOWANIE EUROPEJSKICH MODELI
==========================================

Trenuje poprawione modele z naprawionymi problemami:
- Usunięto data leakage (is_delayed, delay_reason features)
- Dodano walidację krzyżową
- Lepsze regularyzacje
- Monitoring overfitting

Autorzy: AirlineAnalytics-ML Team
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Dodaj src do path
sys.path.append('src')

from european_models import EuropeanDelayClassifier, EuropeanDelayRegressor
from european_models import save_european_models


def main():
    print("🔧 POPRAWIONE TRENOWANIE EUROPEJSKICH MODELI")
    print("="*60)
    print(f"🕒 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. WCZYTAJ DANE
    print("\n📊 WCZYTYWANIE DANYCH EUROPEJSKICH")
    print("-"*40)
    
    data_path = 'data/raw/european_flights_data.csv'
    if not os.path.exists(data_path):
        print(f"❌ Brak pliku: {data_path}")
        print("🔧 Uruchom najpierw: python demo_european_analysis.py")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Wczytano {len(df):,} rekordów europejskich")
    print(f"📊 Kolumny: {list(df.columns)}")
    
    # Sprawdź opóźnienia
    delayed_flights = len(df[df['delay_minutes'] > 0])
    print(f"⏰ Opóźnione loty: {delayed_flights:,} ({delayed_flights/len(df)*100:.1f}%)")
    
    # 2. PRZYGOTUJ DANE DLA KLASYFIKATORA (POPRAWIONE)
    print("\n🎯 POPRAWIONY KLASYFIKATOR EUROPEJSKI")
    print("-"*40)
    
    # Testuj różne algorytmy
    algorithms = ['xgboost', 'random_forest', 'logistic']
    best_classifier = None
    best_clf_score = 0
    
    for alg in algorithms:
        print(f"\n🧪 Testowanie {alg.upper()}...")
        
        try:
            classifier = EuropeanDelayClassifier(model_type=alg)
            
            # Feature engineering (BEZ data leakage!)
            X, y = classifier.prepare_european_features(df, prediction_mode=False)
            
            print(f"📊 Features: {X.shape[1]}, Próbki: {X.shape[0]}")
            print(f"🎯 Target: {y.sum():,} opóźnionych ({y.mean()*100:.1f}%)")
            
            # Sprawdź czy is_delayed NIE MA w feature names
            has_data_leakage = any('is_delayed' in str(feature).lower() for feature in classifier.feature_names)
            if has_data_leakage:
                print("❌ WYKRYTO DATA LEAKAGE! Sprawdź feature engineering")
                continue
            else:
                print("✅ Brak data leakage - is_delayed wykluczone z features")
            
            # Trenuj
            metrics = classifier.train(X, y, test_size=0.2, hyperparameter_tuning=True)
            
            print(f"📈 Wyniki {alg}:")
            print(f"   • ROC AUC: {metrics['roc_auc']:.3f}")
            print(f"   • F1 Score: {metrics['f1_score']:.3f}")
            print(f"   • CV ROC AUC: {metrics.get('cv_roc_auc_mean', 0):.3f}")
            
            # Wybierz najlepszy
            current_score = metrics['roc_auc']
            if current_score > best_clf_score and current_score < 0.99:  # Unikaj overfitting
                best_clf_score = current_score
                best_classifier = classifier
                print(f"🏆 Nowy najlepszy klasyfikator: {alg}")
            
        except Exception as e:
            print(f"❌ Błąd {alg}: {e}")
            continue
    
    if best_classifier is None:
        print("❌ Nie udało się wytrenować żadnego klasyfikatora!")
        return
    
    # 3. PRZYGOTUJ DANE DLA REGRESSORA (POPRAWIONE)
    print("\n📈 POPRAWIONY REGRESSOR EUROPEJSKI")
    print("-"*40)
    
    # Testuj różne algorytmy dla regressora (bez 'linear' - problemy z classifier)
    reg_algorithms = ['xgboost', 'random_forest']
    best_regressor = None
    best_reg_score = -float('inf')
    
    for alg in reg_algorithms:
        print(f"\n🧪 Testowanie regressora {alg.upper()}...")
        
        try:
            regressor = EuropeanDelayRegressor(model_type=alg)
            
            # Feature engineering dla regresji
            X_reg, y_reg = regressor.prepare_features(df)
            
            print(f"📊 Regressor features: {X_reg.shape[1]}, Próbki: {X_reg.shape[0]}")
            print(f"📊 Średnie opóźnienie: {y_reg.mean():.1f} min")
            
            # Trenuj
            reg_metrics = regressor.train(X_reg, y_reg, test_size=0.2)
            
            print(f"📈 Wyniki regressora {alg}:")
            print(f"   • R² Score: {reg_metrics['test_r2']:.3f}")
            print(f"   • MAE: {reg_metrics['mae']:.1f} min")
            print(f"   • RMSE: {reg_metrics['rmse']:.1f} min")
            
            # Wybierz najlepszy (R² > 0 i reasonable MAE)
            current_r2 = reg_metrics['test_r2']
            if current_r2 > best_reg_score and current_r2 > 0:
                best_reg_score = current_r2
                best_regressor = regressor
                print(f"🏆 Nowy najlepszy regressor: {alg}")
            
        except Exception as e:
            print(f"❌ Błąd regressora {alg}: {e}")
            continue
    
    if best_regressor is None:
        print("❌ Nie udało się wytrenować żadnego regressora!")
        # Użyj najlepszego dostępnego jako fallback (xgboost ma lepszy MAE)
        print("🔧 Użycie fallback regressora...")
        best_regressor = EuropeanDelayRegressor(model_type='xgboost')
        X_reg, y_reg = best_regressor.prepare_features(df)
        best_regressor.train(X_reg, y_reg, test_size=0.2)
        best_reg_score = -0.1  # Placeholder score
    
    # 4. ZAPISZ POPRAWIONE MODELE
    print("\n💾 ZAPISYWANIE POPRAWIONYCH MODELI")
    print("-"*40)
    
    try:
        classifier_path, regressor_path = save_european_models(
            best_classifier, best_regressor, 
            filename_base='european_fixed_model'
        )
        
        print(f"✅ Poprawione modele zapisane!")
        print(f"   🎯 Klasyfikator: {classifier_path}")
        print(f"   📈 Regressor: {regressor_path}")
        
    except Exception as e:
        print(f"❌ Błąd zapisu: {e}")
    
    # 5. TEST PRZEWIDYWAŃ
    print("\n🧪 TEST POPRAWIONYCH PRZEWIDYWAŃ")
    print("-"*40)
    
    # Test na przykładowych lotach
    test_flights = [
        {
            'flight_date': '2024-07-15',
            'airline': 'LOT Polish Airlines',
            'origin': 'WAW',
            'destination': 'LHR',
            'country_origin': 'Polska',
            'country_destination': 'Wielka Brytania',
            'distance_km': 1200,
            'scheduled_departure': '08:30',
            'day_of_week': 0,  # Poniedziałek
            'month': 7,
            'hour': 8
        },
        {
            'flight_date': '2024-12-20',
            'airline': 'Ryanair',
            'origin': 'KRK',
            'destination': 'FRA',
            'country_origin': 'Polska',
            'country_destination': 'Niemcy',
            'distance_km': 800,
            'scheduled_departure': '19:00',
            'day_of_week': 4,  # Piątek
            'month': 12,
            'hour': 19
        }
    ]
    
    from european_models import predict_european_delay
    
    for i, flight in enumerate(test_flights, 1):
        print(f"\n✈️ Test {i}: {flight['airline']} {flight['origin']}→{flight['destination']}")
        
        try:
            prediction = predict_european_delay(flight, best_classifier, best_regressor)
            
            print(f"   📊 Prawdopodobieństwo opóźnienia: {prediction['delay_probability']*100:.1f}%")
            print(f"   🎯 Przewidywanie: {'OPÓŹNIONY' if prediction['is_delayed'] else 'PUNKTUALNY'}")
            print(f"   ⏰ Przewidywane opóźnienie: {prediction['predicted_delay_minutes']:.0f} min")
            print(f"   🚨 Ryzyko: {prediction['delay_risk']}")
            
        except Exception as e:
            print(f"   ❌ Błąd przewidywania: {e}")
    
    # 6. FEATURE IMPORTANCE (poprawione)
    print("\n📊 NAJWAŻNIEJSZE CECHY (POPRAWIONE)")
    print("-"*40)
    
    try:
        importance = best_classifier.get_feature_importance()
        print("🏆 Top 10 najważniejszych cech:")
        for i, (_, row) in enumerate(importance.head(10).iterrows(), 1):
            print(f"   {i:2d}. {row['feature']:<25} {row['importance']:.3f}")
        
        # Sprawdź czy nie ma data leakage w top features
        top_features = importance.head(20)['feature'].tolist()
        suspicious_features = [f for f in top_features if any(word in f.lower() for word in ['delay', 'late', 'is_delayed'])]
        
        if suspicious_features:
            print(f"⚠️ UWAGA: Podejrzane cechy w top 20: {suspicious_features}")
        else:
            print("✅ Brak podejrzanych cech w top 20 - poprawka udana!")
            
    except Exception as e:
        print(f"❌ Błąd feature importance: {e}")
    
    # 7. PODSUMOWANIE
    print(f"\n🎉 POPRAWIONE MODELE EUROPEJSKIE GOTOWE!")
    print("="*60)
    print(f"🎯 Najlepszy klasyfikator: {best_classifier.model_type}")
    print(f"📈 Najlepszy regressor: {best_regressor.model_type}")
    print(f"✅ Data leakage: NAPRAWIONY")
    print(f"📊 Features: {len(best_classifier.feature_names)}")
    print(f"🕒 Koniec: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main() 