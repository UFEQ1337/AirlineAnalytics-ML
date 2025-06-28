"""
🇪🇺 MODELE MACHINE LEARNING - DANE EUROPEJSKIE (POPRAWIONA WERSJA)
================================================================

Poprawiona wersja modeli europejskich:
- Usunięto data leakage (is_delayed z feature set)
- Poprawiono feature engineering
- Dodano regularyzację i walidację krzyżową
- Ulepszone hyperparametry

Autorzy: AirlineAnalytics-ML Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, mean_squared_error, mean_absolute_error, r2_score
)

import xgboost as xgb
import joblib
import shap
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')


class EuropeanDelayClassifierFixed:
    """
    🇪🇺 Poprawiony klasyfikator opóźnień - wersja europejska
    Przewiduje czy lot będzie opóźniony (>15 min) bez data leakage
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """Inicjalizacja poprawionego klasyfikatora europejskiego"""
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.label_encoders = {}
        
        # Europejskie mapy
        self.european_regions = self._create_european_regions_map()
        self.problematic_airports = ['LHR', 'CDG', 'FRA', 'AMS']
        self.polish_airports = ['WAW', 'KRK', 'GDN', 'WRO', 'KTW', 'POZ', 'RZE', 'LUZ']
        self.low_cost_carriers = ['Ryanair', 'Wizz Air', 'easyJet', 'Norwegian']
        
        # Inicjalizacja modelu z regularyzacją
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1,
                max_depth=10, min_samples_split=5, min_samples_leaf=2
            )
        elif model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                random_state=42, eval_metric='logloss',
                max_depth=6, learning_rate=0.1, n_estimators=100,
                subsample=0.8, colsample_bytree=0.8
            )
        else:
            raise ValueError("Dostępne modele: 'logistic', 'random_forest', 'xgboost'")
    
    def _create_european_regions_map(self) -> Dict[str, str]:
        """Mapowanie krajów do regionów europejskich"""
        return {
            'Polska': 'Central_Europe',
            'Niemcy': 'Central_Europe', 
            'Czechy': 'Central_Europe',
            'Węgry': 'Central_Europe',
            'Austria': 'Central_Europe',
            'Szwajcaria': 'Central_Europe',
            
            'Wielka Brytania': 'Western_Europe',
            'Francja': 'Western_Europe',
            'Holandia': 'Western_Europe',
            'Belgia': 'Western_Europe',
            
            'Hiszpania': 'Southern_Europe',
            'Włochy': 'Southern_Europe',
            'Portugalia': 'Southern_Europe',
            
            'Szwecja': 'Northern_Europe',
            'Norwegia': 'Northern_Europe',
            'Dania': 'Northern_Europe',
            'Finlandia': 'Northern_Europe',
            'Islandia': 'Northern_Europe',
            
            'Litwa': 'Baltic',
            'Łotwa': 'Baltic', 
            'Estonia': 'Baltic'
        }
    
    def prepare_european_features_fixed(self, df: pd.DataFrame, prediction_mode: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        🔧 POPRAWIONY Feature Engineering - BEZ DATA LEAKAGE
        
        Args:
            df (DataFrame): Europejskie dane wejściowe
            prediction_mode (bool): Czy przygotowujemy dane do przewidywania
            
        Returns:
            Tuple[DataFrame, Series]: Przetworzone cechy i target
        """
        print("🔧 POPRAWIONY EUROPEAN FEATURE ENGINEERING")
        print("="*60)
        
        df_features = df.copy()
        
        # 1. TARGET VARIABLE - TYLKO w trybie treningu
        target = None
        if not prediction_mode and 'delay_minutes' in df_features.columns:
            target = (df_features['delay_minutes'] > 15).astype(int)
            print(f"🎯 Target: {target.sum():,} opóźnionych z {len(target):,} ({target.mean()*100:.1f}%)")
        
        # 2. TEMPORAL FEATURES (europejskie)
        print("⏰ Europejskie cechy czasowe...")
        df_features['flight_date'] = pd.to_datetime(df_features['flight_date'])
        
        # Basic temporal
        if 'hour' not in df_features.columns:
            df_features['hour'] = pd.to_datetime(df_features['scheduled_departure'], format='%H:%M').dt.hour
        if 'day_of_week' not in df_features.columns:
            df_features['day_of_week'] = df_features['flight_date'].dt.dayofweek
        if 'month' not in df_features.columns:
            df_features['month'] = df_features['flight_date'].dt.month
            
        df_features['day_of_month'] = df_features['flight_date'].dt.day
        df_features['quarter'] = df_features['flight_date'].dt.quarter
        df_features['week_of_year'] = df_features['flight_date'].dt.isocalendar().week
        
        # European specific temporal features
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        df_features['is_summer_holiday'] = df_features['month'].isin([7, 8]).astype(int)
        df_features['is_winter_season'] = df_features['month'].isin([12, 1, 2]).astype(int)
        df_features['is_spring_autumn'] = df_features['month'].isin([3, 4, 9, 10]).astype(int)
        
        # European rush hours
        df_features['is_morning_rush_eu'] = df_features['hour'].isin([6, 7, 8]).astype(int)
        df_features['is_evening_rush_eu'] = df_features['hour'].isin([18, 19, 20]).astype(int)
        df_features['is_business_hours'] = df_features['hour'].isin([9, 10, 11, 14, 15, 16]).astype(int)
        df_features['is_late_night'] = df_features['hour'].isin([22, 23, 0, 1]).astype(int)
        
        # Cyclical encoding
        print("🔄 Cyclical encoding...")
        for col, max_val in [('hour', 24), ('day_of_week', 7), ('month', 12)]:
            df_features[f'{col}_sin'] = np.sin(2 * np.pi * df_features[col] / max_val)
            df_features[f'{col}_cos'] = np.cos(2 * np.pi * df_features[col] / max_val)
        
        # 3. EUROPEAN GEOGRAPHIC FEATURES
        print("🌍 Europejskie cechy geograficzne...")
        
        # Region mapping
        if 'country_origin' in df_features.columns:
            df_features['region_origin'] = df_features['country_origin'].map(self.european_regions).fillna('Other')
            df_features['region_destination'] = df_features['country_destination'].map(self.european_regions).fillna('Other')
            
            # Cross-region flights
            df_features['cross_region_flight'] = (df_features['region_origin'] != df_features['region_destination']).astype(int)
            
            # Polish connections
            df_features['from_poland'] = (df_features['country_origin'] == 'Polska').astype(int)
            df_features['to_poland'] = (df_features['country_destination'] == 'Polska').astype(int)
            df_features['polish_connection'] = (df_features['from_poland'] | df_features['to_poland']).astype(int)
        
        # Airport-specific features
        if 'origin' in df_features.columns:
            df_features['problematic_origin'] = df_features['origin'].isin(self.problematic_airports).astype(int)
            df_features['problematic_destination'] = df_features['destination'].isin(self.problematic_airports).astype(int)
            df_features['polish_origin'] = df_features['origin'].isin(self.polish_airports).astype(int)
            df_features['polish_destination'] = df_features['destination'].isin(self.polish_airports).astype(int)
        
        # 4. DISTANCE FEATURES
        distance_col = 'distance_km' if 'distance_km' in df_features.columns else 'distance_miles'
        if distance_col in df_features.columns:
            df_features['distance_normalized'] = df_features[distance_col] / df_features[distance_col].max()
            
            # European distance categories
            if distance_col == 'distance_km':
                bins = [0, 500, 1000, 1500, float('inf')]
                labels = ['Short_EU', 'Medium_EU', 'Long_EU', 'Ultra_Long_EU']
            else:
                bins = [0, 310, 620, 930, float('inf')]  # Mile equivalent
                labels = ['Short_EU', 'Medium_EU', 'Long_EU', 'Ultra_Long_EU']
            
            df_features['distance_category'] = pd.cut(df_features[distance_col], bins=bins, labels=labels)
            
            # Distance indicators
            threshold = 800 if distance_col == 'distance_km' else 500
            df_features['short_eu_flight'] = (df_features[distance_col] < threshold).astype(int)
        
        # 5. EUROPEAN CARRIER FEATURES
        print("✈️ Cechy przewoźników europejskich...")
        if 'airline' in df_features.columns:
            df_features['is_low_cost'] = df_features['airline'].isin(self.low_cost_carriers).astype(int)
            df_features['is_lot'] = (df_features['airline'] == 'LOT Polish Airlines').astype(int)
            df_features['is_lufthansa_group'] = df_features['airline'].isin(['Lufthansa', 'Austrian Airlines', 'Swiss', 'Eurowings']).astype(int)
            df_features['is_air_france_klm'] = df_features['airline'].isin(['Air France', 'KLM']).astype(int)
        
        # 6. HISTORICAL DELAY PATTERNS (bez data leakage)
        print("📈 Wzorce historyczne (bez data leakage)...")
        # Te cechy bazują na wzorcach, ale nie na rzeczywistych opóźnieniach tego lotu
        if 'hour' in df_features.columns:
            # Rush hour risk (statystyczny)
            df_features['rush_hour_risk'] = df_features['hour'].map({
                6: 0.3, 7: 0.4, 8: 0.5, 9: 0.3,  # Poranny rush
                17: 0.3, 18: 0.4, 19: 0.5, 20: 0.4  # Wieczorny rush
            }).fillna(0.2)
        
        # Seasonal risk
        if 'month' in df_features.columns:
            df_features['seasonal_risk'] = df_features['month'].map({
                12: 0.4, 1: 0.5, 2: 0.4,  # Zima - wyższe ryzyko
                3: 0.2, 4: 0.2, 5: 0.2,   # Wiosna - niższe ryzyko
                6: 0.3, 7: 0.3, 8: 0.3,   # Lato - średnie ryzyko
                9: 0.2, 10: 0.2, 11: 0.3  # Jesień - niższe ryzyko
            }).fillna(0.3)
        
        # 7. ONE-HOT ENCODING (ograniczone)
        print("🏷️ One-hot encoding (top kategorie)...")
        
        # Top airlines (tylko 8 najbardziej popularnych)
        if 'airline' in df_features.columns:
            top_airlines = df_features['airline'].value_counts().head(8).index
            for airline in top_airlines:
                df_features[f'airline_{airline.replace(" ", "_").replace("-", "_")}'] = (df_features['airline'] == airline).astype(int)
        
        # Top origin airports (tylko 10 najbardziej popularnych)
        if 'origin' in df_features.columns:
            top_origins = df_features['origin'].value_counts().head(10).index
            for origin in top_origins:
                df_features[f'origin_{origin}'] = (df_features['origin'] == origin).astype(int)
        
        # Regions
        if 'region_origin' in df_features.columns:
            for region in ['Central_Europe', 'Western_Europe', 'Northern_Europe', 'Southern_Europe']:
                df_features[f'region_orig_{region}'] = (df_features['region_origin'] == region).astype(int)
                df_features[f'region_dest_{region}'] = (df_features['region_destination'] == region).astype(int)
        
        # 8. SELECT FINAL FEATURES (BEZ data leakage)
        print("🔍 Selekcja finalnych cech (bez data leakage)...")
        
        # WYKLUCZAMY wszystkie kolumny które mogą zawierać data leakage
        exclude_cols = [
            'flight_date', 'scheduled_departure', 'actual_departure', 
            'delay_minutes', 'delay_reason', 'delay_category',  # Te zawierają info o opóźnieniu
            'is_delayed',  # TO JEST NASZ TARGET - MUSI BYĆ WYKLUCZONE!
            'route', 'airline', 'origin', 'destination',
            'aircraft_type', 'country_origin', 'country_destination', 
            'origin_city', 'destination_city',
            'region_origin', 'region_destination', 'distance_category',
            'day_of_month', 'quarter', 'week_of_year'  # Mniej istotne cechy czasowe
        ]
        
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        X = df_features[feature_cols].select_dtypes(include=[np.number])
        
        # Handle missing values
        X = X.fillna(0)
        
        # Remove constant features
        constant_features = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_features:
            print(f"🗑️ Usuwanie {len(constant_features)} stałych cech")
            X = X.drop(columns=constant_features)
        
        # Save feature names
        self.feature_names = list(X.columns)
        print(f"✅ Poprawiony feature engineering zakończony!")
        print(f"📊 Utworzono {len(self.feature_names)} cech (bez data leakage)")
        print(f"📝 Przykład cech: {self.feature_names[:8]}")
        print(f"🇵🇱 Cechy polskie: {[f for f in self.feature_names if 'polish' in f.lower() or 'poland' in f.lower()]}")
        print(f"✅ BRAK is_delayed w feature set: {'is_delayed' not in self.feature_names}")
        
        if target is not None:
            return X, target
        else:
            return X, None
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, hyperparameter_tuning: bool = True):
        """🎯 Trenowanie poprawionego modelu z walidacją krzyżową"""
        print(f"🔧 TRENOWANIE POPRAWIONEGO MODELU: {self.model_type.upper()}")
        print("="*60)
        
        # Convert to arrays
        X_array = X.values if hasattr(X, 'values') else np.array(X)
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        
        # Train/test split - stratified
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array, test_size=test_size, random_state=42, stratify=y_array
        )
        
        print(f"📊 Podział danych:")
        print(f"   • Train: {X_train.shape[0]:,} próbek ({y_train.mean()*100:.1f}% opóźnionych)")
        print(f"   • Test:  {X_test.shape[0]:,} próbek ({y_test.mean()*100:.1f}% opóźnionych)")
        print(f"   • Features: {X_train.shape[1]}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Cross-validation przed tuningiem
        print("🔄 Walidacja krzyżowa baseline...")
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"   📊 CV ROC AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            print("🔧 Optymalizacja hiperparametrów...")
            self._tune_fixed_hyperparameters(X_train_scaled, y_train)
        
        # Train model
        print("🚀 Trenowanie modelu...")
        self.model.fit(X_train_scaled, y_train)
        
        # Store test data
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"✅ Trenowanie zakończone!")
        print(f"📈 Accuracy (train): {train_score:.3f}")
        print(f"📈 Accuracy (test):  {test_score:.3f}")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        
        roc_auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"🎯 Metryki szczegółowe:")
        print(f"   • ROC AUC: {roc_auc:.3f}")
        print(f"   • Precision: {precision:.3f}")
        print(f"   • Recall: {recall:.3f}")
        print(f"   • F1 Score: {f1:.3f}")
        
        # Final cross-validation
        final_cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"📊 Final CV ROC AUC: {final_cv_scores.mean():.3f} ± {final_cv_scores.std():.3f}")
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_roc_auc_mean': final_cv_scores.mean(),
            'cv_roc_auc_std': final_cv_scores.std()
        }
    
    def _tune_fixed_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """Poprawiona optymalizacja hiperparametrów"""
        
        if self.model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        elif self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'logistic':
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs']
            }
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring='roc_auc', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        print(f"   🏆 Najlepsze parametry: {grid_search.best_params_}")
        print(f"   📊 Najlepszy wynik CV: {grid_search.best_score_:.3f}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Przewidywanie prawdopodobieństwa opóźnienia"""
        X_scaled = self.scaler.transform(X.values if hasattr(X, 'values') else X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict_binary(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Przewidywanie binarnego wyniku"""
        X_scaled = self.scaler.transform(X.values if hasattr(X, 'values') else X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Zwraca ważność cech"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            return None
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance


class EuropeanDelayRegressorFixed:
    """🇪🇺 Poprawiony regressor opóźnień - wersja europejska"""
    
    def __init__(self, model_type: str = 'xgboost'):
        """Inicjalizacja poprawionego regressora"""
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.classifier = EuropeanDelayClassifierFixed(model_type)
        
        # Inicjalizacja modelu z regularyzacją
        if model_type == 'linear':
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1,
                max_depth=10, min_samples_split=5, min_samples_leaf=2
            )
        elif model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                random_state=42,
                max_depth=6, learning_rate=0.1, n_estimators=100,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1  # L1 i L2 regularization
            )
        else:
            raise ValueError("Dostępne modele: 'linear', 'random_forest', 'xgboost'")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Przygotowanie cech dla regresji (tylko opóźnione loty)"""
        print("🔧 POPRAWIONY EUROPEAN REGRESSOR - FEATURE ENGINEERING")
        print("="*60)
        
        # Filtruj tylko opóźnione loty (>0 min)
        delayed_flights = df[df['delay_minutes'] > 0].copy()
        print(f"📊 Loty opóźnione: {len(delayed_flights):,} z {len(df):,} ({len(delayed_flights)/len(df)*100:.1f}%)")
        
        # Usuń ekstremalne outliers (>4 godziny opóźnienia)
        before_outliers = len(delayed_flights)
        delayed_flights = delayed_flights[delayed_flights['delay_minutes'] <= 240]
        print(f"🗑️ Usunięto {before_outliers - len(delayed_flights)} outliers (>4h opóźnienia)")
        
        # Użyj poprawionego feature engineering
        X, _ = self.classifier.prepare_european_features_fixed(delayed_flights, prediction_mode=True)
        y = delayed_flights['delay_minutes']
        
        self.feature_names = self.classifier.feature_names
        
        print(f"✅ Regressor features: {X.shape[1]} cech, {len(y)} próbek")
        print(f"📊 Średnie opóźnienie: {y.mean():.1f} min, mediana: {y.median():.1f} min")
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Trenowanie poprawionego modelu regresji"""
        print(f"🔧 TRENOWANIE POPRAWIONEGO REGRESSORA: {self.model_type.upper()}")
        print("="*60)
        
        X_array = X.values if hasattr(X, 'values') else np.array(X)
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array, test_size=test_size, random_state=42
        )
        
        print(f"📊 Podział danych:")
        print(f"   • Train: {X_train.shape[0]:,} próbek (średnie: {y_train.mean():.1f} min)")
        print(f"   • Test:  {X_test.shape[0]:,} próbek (średnie: {y_test.mean():.1f} min)")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Cross-validation
        print("🔄 Walidacja krzyżowa...")
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"   📊 CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Train model
        print("🚀 Trenowanie modelu...")
        self.model.fit(X_train_scaled, y_train)
        
        # Store test data
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"✅ Trenowanie zakończone!")
        print(f"📈 R² (train): {train_r2:.3f}")
        print(f"📈 R² (test):  {test_r2:.3f}")
        print(f"📊 MAE (test): {test_mae:.1f} min")
        print(f"📊 RMSE (test): {test_rmse:.1f} min")
        
        # Final cross-validation
        final_cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"📊 Final CV R²: {final_cv_scores.mean():.3f} ± {final_cv_scores.std():.3f}")
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'mae': test_mae,
            'rmse': test_rmse,
            'cv_r2_mean': final_cv_scores.mean(),
            'cv_r2_std': final_cv_scores.std()
        }
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Przewidywanie liczby minut opóźnienia"""
        X_scaled = self.scaler.transform(X.values if hasattr(X, 'values') else X)
        predictions = self.model.predict(X_scaled)
        return np.maximum(0, predictions)  # Nie może być ujemnych opóźnień


def save_european_models_fixed(classifier: EuropeanDelayClassifierFixed, 
                              regressor: EuropeanDelayRegressorFixed, 
                              filename_base: str = 'european_fixed_model'):
    """Zapisuje poprawione europejskie modele"""
    
    classifier_path = f'notebooks/{filename_base}_classifier.joblib'
    regressor_path = f'notebooks/{filename_base}_regressor.joblib'
    
    joblib.dump(classifier, classifier_path)
    joblib.dump(regressor, regressor_path)
    
    print(f"✅ Poprawione europejskie modele zapisane:")
    print(f"   🎯 Klasyfikator: {classifier_path}")
    print(f"   📈 Regressor: {regressor_path}")
    
    return classifier_path, regressor_path


def predict_european_delay_fixed(flight_details: Dict[str, Any], 
                                 classifier: EuropeanDelayClassifierFixed, 
                                 regressor: EuropeanDelayRegressorFixed) -> Dict[str, Any]:
    """Przewidywanie opóźnienia dla lotu europejskiego (poprawiona wersja)"""
    
    try:
        # Stwórz DataFrame z danymi lotu
        df_flight = pd.DataFrame([flight_details])
        
        # Feature engineering (BEZ data leakage)
        X_clf, _ = classifier.prepare_european_features_fixed(df_flight, prediction_mode=True)
        
        # Przewidywanie klasyfikacji
        delay_probability = classifier.predict(X_clf)[0]
        is_delayed = classifier.predict_binary(X_clf)[0]
        
        result = {
            'delay_probability': delay_probability,
            'is_delayed': bool(is_delayed),
            'delay_risk': 'Wysokie' if delay_probability > 0.7 else 'Średnie' if delay_probability > 0.3 else 'Niskie'
        }
        
        # Jeśli przewidujemy opóźnienie, oszacuj minuty
        if is_delayed:
            delay_minutes = regressor.predict(X_clf)[0]
            result['predicted_delay_minutes'] = max(0, delay_minutes)
            result['delay_category'] = 'Małe' if delay_minutes < 30 else 'Średnie' if delay_minutes < 60 else 'Duże'
        else:
            result['predicted_delay_minutes'] = 0
            result['delay_category'] = 'Punktualny'
        
        return result
        
    except Exception as e:
        print(f"❌ Błąd przewidywania: {e}")
        return {
            'error': str(e),
            'delay_probability': 0.0,
            'is_delayed': False,
            'predicted_delay_minutes': 0
        } 