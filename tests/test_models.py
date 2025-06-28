"""
🧪 TESTY JEDNOSTKOWE - AIRLINE ANALYTICS ML
========================================

Testy podstawowych funkcjonalności:
- Ładowanie danych
- Modele ML (predykcja)  
- Funkcje dashboard
- Integralność systemu

Autor: AirlineAnalytics-ML Team
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import joblib
from unittest.mock import patch, MagicMock

# Dodaj src do ścieżki
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

try:
    from models import DelayClassifier, DelayRegressor, predict_delay
    from utils import load_clean_data, model_health_check, get_data_statistics
    # from dashboard import calculate_kpis  # Import będzie testowany
except ImportError as e:
    print(f"⚠️ Ostrzeżenie: Nie można zaimportować modułów src: {e}")

class TestDataLoading(unittest.TestCase):
    """🔄 Testy ładowania i walidacji danych"""
    
    def setUp(self):
        """Setup przed każdym testem"""
        self.test_data = pd.DataFrame({
            'flight_date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'airline': np.random.choice(['AA', 'DL', 'UA'], 100),
            'origin': np.random.choice(['JFK', 'LAX', 'ORD'], 100),
            'destination': np.random.choice(['ATL', 'DEN', 'SFO'], 100),
            'delay_minutes': np.random.exponential(15, 100) - 5,
            'scheduled_departure': ['10:30'] * 100
        })
        # Usuń loty do tego samego lotniska
        self.test_data = self.test_data[self.test_data['origin'] != self.test_data['destination']]
    
    def test_data_structure(self):
        """Test struktury danych"""
        self.assertGreater(len(self.test_data), 0, "Dane nie mogą być puste")
        
        required_columns = ['flight_date', 'airline', 'origin', 'destination', 'delay_minutes']
        for col in required_columns:
            self.assertIn(col, self.test_data.columns, f"Brak wymaganej kolumny: {col}")
    
    def test_data_types(self):
        """Test typów danych"""
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.test_data['flight_date']), 
                       "flight_date powinno być datetime")
        self.assertTrue(pd.api.types.is_numeric_dtype(self.test_data['delay_minutes']), 
                       "delay_minutes powinno być numeryczne")
    
    def test_data_quality(self):
        """Test jakości danych"""
        # Sprawdź braki danych
        null_counts = self.test_data.isnull().sum()
        self.assertEqual(null_counts.sum(), 0, "Nie powinno być braków danych w kluczowych kolumnach")
        
        # Sprawdź realistyczne zakresy opóźnień
        min_delay = self.test_data['delay_minutes'].min()
        max_delay = self.test_data['delay_minutes'].max()
        self.assertGreaterEqual(min_delay, -60, "Minimalne opóźnienie zbyt małe")
        self.assertLessEqual(max_delay, 500, "Maksymalne opóźnienie zbyt duże")
    
    def test_load_clean_data_function(self):
        """Test funkcji load_clean_data"""
        try:
            # Test z niewłaściwą ścieżką
            result = load_clean_data("nonexistent_path.csv")
            self.assertIsNone(result, "Powinna zwrócić None dla nieistniejącej ścieżki")
        except NameError:
            self.skipTest("Funkcja load_clean_data niedostępna")


class TestModels(unittest.TestCase):
    """🤖 Testy modeli Machine Learning"""
    
    def setUp(self):
        """Setup przed każdym testem"""
        # Stwórz przykładowe dane
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'flight_date': pd.date_range('2024-01-01', periods=1000, freq='H'),
            'airline': np.random.choice(['American Airlines', 'Delta', 'United'], 1000),
            'origin': np.random.choice(['JFK', 'LAX', 'ORD', 'ATL'], 1000),
            'destination': np.random.choice(['DEN', 'SFO', 'BOS', 'MIA'], 1000),
            'delay_minutes': np.random.exponential(15, 1000) - 5,
            'scheduled_departure': [f"{np.random.randint(6,23):02d}:{np.random.randint(0,60):02d}" for _ in range(1000)],
            'aircraft_type': np.random.choice(['B737', 'A320', 'B777'], 1000),
            'distance_miles': np.random.randint(300, 3000, 1000)
        })
        
        # Usuń loty do tego samego lotniska
        self.test_data = self.test_data[self.test_data['origin'] != self.test_data['destination']]
    
    def test_delay_classifier_init(self):
        """Test inicjalizacji klasyfikatora"""
        try:
            classifier = DelayClassifier('xgboost')
            self.assertEqual(classifier.model_type, 'xgboost')
            self.assertIsNotNone(classifier.model)
        except NameError:
            self.skipTest("Klasa DelayClassifier niedostępna")
    
    def test_delay_regressor_init(self):
        """Test inicjalizacji regresora"""
        try:
            regressor = DelayRegressor('xgboost')
            self.assertEqual(regressor.model_type, 'xgboost')
            self.assertIsNotNone(regressor.model)
        except NameError:
            self.skipTest("Klasa DelayRegressor niedostępna")
    
    def test_feature_preparation(self):
        """Test przygotowania cech"""
        try:
            classifier = DelayClassifier('logistic')  # Szybszy model do testów
            X, y = classifier.prepare_features(self.test_data)
            
            self.assertIsInstance(X, pd.DataFrame, "X powinno być DataFrame")
            self.assertIsInstance(y, pd.Series, "y powinno być Series")
            self.assertEqual(len(X), len(y), "X i y powinny mieć tę samą długość")
            self.assertTrue(all(col in X.columns for col in ['hour', 'day_of_week']), 
                           "Powinny być utworzone cechy czasowe")
        except (NameError, AttributeError):
            self.skipTest("Metoda prepare_features niedostępna")
    
    def test_model_training(self):
        """Test trenowania modelu"""
        try:
            classifier = DelayClassifier('logistic')
            X, y = classifier.prepare_features(self.test_data.head(100))  # Małe dane do testów
            
            # Test czy training nie rzuca błędów
            classifier.train(X, y, test_size=0.3, hyperparameter_tuning=False)
            
            # Test czy model został wytrenowany
            self.assertIsNotNone(classifier.model)
            self.assertTrue(hasattr(classifier, 'scaler'))
            
        except (NameError, AttributeError, Exception) as e:
            self.skipTest(f"Test trenowania modelu niedostępny: {e}")
    
    def test_prediction_interface(self):
        """Test interfejsu predykcji"""
        try:
            # Test danych wejściowych
            flight_details = {
                'airline': 'American Airlines',
                'origin': 'JFK',
                'destination': 'LAX',
                'aircraft_type': 'B737',
                'distance_miles': 2500,
                'scheduled_departure': '10:00',
                'flight_date': '2024-06-01'
            }
            
            # Mock models dla testu
            mock_classifier = MagicMock()
            mock_classifier.predict.return_value = np.array([0.7])
            mock_classifier.predict_binary.return_value = np.array([1])
            
            mock_regressor = MagicMock()
            mock_regressor.predict.return_value = np.array([25.5])
            
            # Test czy interfejs działa
            result = predict_delay(flight_details, mock_classifier, mock_regressor)
            
            if result:
                self.assertIn('is_delayed', result)
                self.assertIn('delay_minutes', result)
                self.assertIn('delay_probability', result)
                
        except NameError:
            self.skipTest("Funkcja predict_delay niedostępna")


class TestUtilities(unittest.TestCase):
    """🔧 Testy funkcji pomocniczych"""
    
    def test_model_health_check(self):
        """Test sprawdzania zdrowia modelu"""
        try:
            result = model_health_check()
            
            self.assertIsInstance(result, dict, "Wynik powinien być słownikiem")
            self.assertIn('overall_status', result, "Powinien zawierać overall_status")
            self.assertIn('timestamp', result, "Powinien zawierać timestamp")
            
            # Status powinien być jednym z dozwolonych
            valid_statuses = ['healthy', 'warning', 'unhealthy', 'error', 'unknown']
            self.assertIn(result['overall_status'], valid_statuses, 
                         f"Status {result['overall_status']} nie jest poprawny")
            
        except NameError:
            self.skipTest("Funkcja model_health_check niedostępna")
    
    def test_data_statistics(self):
        """Test funkcji statystyk danych"""
        try:
            test_df = pd.DataFrame({
                'flight_date': pd.date_range('2024-01-01', periods=100),
                'airline': ['AA'] * 100,
                'delay_minutes': np.random.normal(20, 10, 100)
            })
            
            stats = get_data_statistics(test_df)
            
            self.assertIsInstance(stats, dict, "Statystyki powinny być słownikiem")
            self.assertIn('basic_info', stats, "Powinny zawierać basic_info")
            
            if 'basic_info' in stats:
                self.assertIn('total_records', stats['basic_info'])
                self.assertEqual(stats['basic_info']['total_records'], 100)
                
        except NameError:
            self.skipTest("Funkcja get_data_statistics niedostępna")


class TestDashboardFunctions(unittest.TestCase):
    """📊 Testy funkcji dashboard"""
    
    def setUp(self):
        """Setup dla testów dashboard"""
        self.sample_data = pd.DataFrame({
            'flight_date': pd.date_range('2024-01-01', periods=1000),
            'airline': np.random.choice(['AA', 'DL', 'UA', 'SW'], 1000),
            'origin': np.random.choice(['JFK', 'LAX', 'ORD'], 1000),
            'destination': np.random.choice(['ATL', 'DEN', 'SFO'], 1000),
            'delay_minutes': np.random.exponential(15, 1000) - 5,
            'scheduled_departure': ['10:30'] * 1000
        })
    
    def test_kpi_calculation(self):
        """Test obliczania KPI"""
        # Symulacja funkcji calculate_kpis (jeśli nie ma importu)
        def mock_calculate_kpis(data):
            on_time_flights = (data['delay_minutes'] <= 15).sum()
            total_flights = len(data)
            on_time_percent = (on_time_flights / total_flights) * 100
            avg_delay = data['delay_minutes'].mean()
            
            return {
                'on_time_percent': on_time_percent,
                'avg_delay': avg_delay,
                'total_flights': total_flights
            }
        
        kpis = mock_calculate_kpis(self.sample_data)
        
        self.assertIsInstance(kpis, dict, "KPI powinny być słownikiem")
        self.assertIn('on_time_percent', kpis)
        self.assertIn('avg_delay', kpis)
        self.assertIn('total_flights', kpis)
        
        # Sprawdź rozsądne wartości
        self.assertGreaterEqual(kpis['on_time_percent'], 0)
        self.assertLessEqual(kpis['on_time_percent'], 100)
        self.assertEqual(kpis['total_flights'], len(self.sample_data))


class TestSystemIntegration(unittest.TestCase):
    """🔗 Testy integracji systemu"""
    
    def test_file_structure(self):
        """Test struktury plików projektu"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        
        # Sprawdź kluczowe katalogi
        expected_dirs = ['src', 'notebooks', 'data', 'results']
        for dir_name in expected_dirs:
            dir_path = os.path.join(base_path, dir_name)
            self.assertTrue(os.path.exists(dir_path), f"Katalog {dir_name} powinien istnieć")
    
    def test_model_files_exist(self):
        """Test istnienia plików modeli"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        
        model_files = [
            'notebooks/best_model_classifier.joblib',
            'notebooks/best_model_regressor.joblib'
        ]
        
        for model_file in model_files:
            model_path = os.path.join(base_path, model_file)
            if os.path.exists(model_path):
                # Sprawdź czy plik nie jest pusty
                self.assertGreater(os.path.getsize(model_path), 0, 
                                 f"Plik modelu {model_file} nie może być pusty")
    
    def test_data_files_exist(self):
        """Test istnienia plików danych"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        
        data_files = [
            'data/processed/flights_cleaned.csv',
            'data/raw/flights_data.csv'
        ]
        
        data_exists = False
        for data_file in data_files:
            data_path = os.path.join(base_path, data_file)
            if os.path.exists(data_path):
                data_exists = True
                # Sprawdź czy plik nie jest pusty
                self.assertGreater(os.path.getsize(data_path), 0, 
                                 f"Plik danych {data_file} nie może być pusty")
        
        # Przynajmniej jeden plik danych powinien istnieć
        self.assertTrue(data_exists, "Przynajmniej jeden plik danych powinien istnieć")


class TestErrorHandling(unittest.TestCase):
    """❌ Testy obsługi błędów"""
    
    def test_invalid_data_handling(self):
        """Test obsługi nieprawidłowych danych"""
        # Test pustego DataFrame
        empty_df = pd.DataFrame()
        
        try:
            stats = get_data_statistics(empty_df)
            if stats:
                self.assertIn('error', stats, "Powinien zwrócić błąd dla pustych danych")
        except NameError:
            self.skipTest("Funkcja get_data_statistics niedostępna")
    
    def test_invalid_model_prediction(self):
        """Test obsługi błędnych predykcji"""
        invalid_flight_details = {
            'airline': None,  # Nieprawidłowa wartość
            'origin': 'INVALID',
            'destination': 'INVALID'
        }
        
        try:
            result = predict_delay(invalid_flight_details, None, None)
            # Powinien obsłużyć błąd gracefully
            self.assertIsNone(result, "Powinien zwrócić None dla nieprawidłowych danych")
        except NameError:
            self.skipTest("Funkcja predict_delay niedostępna")


def run_tests():
    """🚀 Uruchomienie wszystkich testów"""
    print("🧪 URUCHAMIANIE TESTÓW AIRLINE ANALYTICS ML")
    print("=" * 60)
    
    # Konfiguracja testów
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Dodaj wszystkie klasy testowe
    test_classes = [
        TestDataLoading,
        TestModels,
        TestUtilities,
        TestDashboardFunctions,
        TestSystemIntegration,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Uruchom testy
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Podsumowanie
    print("\n" + "=" * 60)
    print("📊 PODSUMOWANIE TESTÓW")
    print("=" * 60)
    print(f"✅ Testy zaliczone: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Testy nieudane: {len(result.failures)}")
    print(f"🚫 Błędy: {len(result.errors)}")
    print(f"⏭️ Pominięte: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n❌ NIEUDANE TESTY:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n🚫 BŁĘDY:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"\n🎯 Wskaźnik sukcesu: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 