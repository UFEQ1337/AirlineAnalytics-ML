"""
🤖 MODUŁ MACHINE LEARNING - PRZEWIDYWANIE OPÓŹNIEŃ LOTÓW
=======================================================

Zawiera klasy i funkcje do:
- Feature engineering
- Modele klasyfikacji (czy lot będzie opóźniony?)
- Modele regresji (o ile minut opóźnienie?)
- Ewaluacja modelu i wizualizacje
- Interpretabilność (SHAP)

Autorzy: AirlineAnalytics-ML Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


class DelayClassifier:
    """
    🎯 Klasyfikator opóźnień lotów - przewiduje czy lot będzie opóźniony (>15 min)
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Inicjalizacja klasyfikatora
        
        Args:
            model_type (str): Typ modelu - 'logistic', 'random_forest', 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.label_encoders = {}
        
        # Inicjalizacja modelu
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'xgboost':
            self.model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        else:
            raise ValueError("Dostępne modele: 'logistic', 'random_forest', 'xgboost'")
    
    def prepare_features(self, df: pd.DataFrame, prediction_mode: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        🔧 Feature Engineering - przygotowanie cech do modelu
        
        Args:
            df (DataFrame): Dane wejściowe
            prediction_mode (bool): Czy przygotowujemy dane do przewidywania (bez target)
            
        Returns:
            Tuple[DataFrame, Series]: Przetworzone cechy i target (lub None w trybie prediction)
        """
        print("🔧 FEATURE ENGINEERING - KLASYFIKACJA")
        print("="*50)
        
        df_features = df.copy()
        
        # 1. TARGET VARIABLE - is_delayed (>15 min) - tylko w trybie treningu
        target = None
        if not prediction_mode and 'delay_minutes' in df_features.columns:
            df_features['is_delayed'] = (df_features['delay_minutes'] > 15).astype(int)
            target = df_features['is_delayed']
        elif prediction_mode:
            # W trybie przewidywania nie mamy target variable
            df_features['is_delayed'] = 0  # Dummy value
        
        # 2. TEMPORAL FEATURES
        print("⏰ Tworzenie cech czasowych...")
        df_features['flight_date'] = pd.to_datetime(df_features['flight_date'])
        
        # Basic temporal features
        df_features['hour'] = pd.to_datetime(df_features['scheduled_departure']).dt.hour
        df_features['day_of_week'] = df_features['flight_date'].dt.dayofweek
        df_features['month'] = df_features['flight_date'].dt.month
        df_features['day_of_month'] = df_features['flight_date'].dt.day
        df_features['quarter'] = df_features['flight_date'].dt.quarter
        
        # Advanced temporal features
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        df_features['is_holiday'] = self._create_holiday_feature(pd.to_datetime(df_features['flight_date']))
        df_features['season'] = df_features['month'].map(self._get_season)
        
        # Rush hour features
        df_features['is_morning_rush'] = df_features['hour'].isin([6, 7, 8, 9]).astype(int)
        df_features['is_evening_rush'] = df_features['hour'].isin([17, 18, 19, 20]).astype(int)
        df_features['is_late_night'] = df_features['hour'].isin([22, 23, 0, 1, 2, 3]).astype(int)
        
        # Cyclical encoding for temporal features
        print("🔄 Cyclical encoding...")
        for col, max_val in [('hour', 24), ('day_of_week', 7), ('month', 12), ('day_of_month', 31)]:
            df_features[f'{col}_sin'] = np.sin(2 * np.pi * df_features[col] / max_val)
            df_features[f'{col}_cos'] = np.cos(2 * np.pi * df_features[col] / max_val)
        
        # 3. CATEGORICAL FEATURES - One-hot encoding
        print("🏷️  One-hot encoding dla kategorii...")
        categorical_cols = ['airline', 'origin', 'destination', 'aircraft_type', 'season']
        
        for col in categorical_cols:
            if col in df_features.columns:
                # Limituj liczbę kategorii (top 10 + 'Other')
                top_categories = df_features[col].value_counts().head(10).index
                df_features[f'{col}_category'] = df_features[col].apply(
                    lambda x: x if x in top_categories else 'Other'
                )
                
                # One-hot encoding
                dummies = pd.get_dummies(df_features[f'{col}_category'], prefix=col, drop_first=True)
                df_features = pd.concat([df_features, dummies], axis=1)
        
        # 4. DISTANCE FEATURES
        if 'distance_miles' in df_features.columns:
            df_features['distance_normalized'] = df_features['distance_miles'] / df_features['distance_miles'].max()
            df_features['distance_category'] = pd.cut(
                df_features['distance_miles'], 
                bins=[0, 500, 1000, 2000, float('inf')],
                labels=['Short', 'Medium', 'Long', 'Ultra_Long']
            )
            # Distance category dummies
            distance_dummies = pd.get_dummies(df_features['distance_category'], prefix='distance', drop_first=True)
            df_features = pd.concat([df_features, distance_dummies], axis=1)
        
        # Wybierz tylko kolumny numeryczne do modelu
        feature_cols = [col for col in df_features.columns if col not in [
            'flight_date', 'scheduled_departure', 'actual_departure', 'delay_minutes',
            'delay_reason', 'delay_category', 'route', 'airline', 'origin', 'destination',
            'aircraft_type', 'season', 'distance_category', 'airline_category',
            'origin_category', 'destination_category', 'aircraft_type_category', 'season_category']]
        
        X = df_features[feature_cols].select_dtypes(include=[np.number])
        
        # Zapisz nazwy cech - WAŻNE!
        self.feature_names = list(X.columns)
        print(f"✅ Feature engineering zakończony! Utworzono {len(self.feature_names)} cech")
        print(f"📝 Przykładowe nazwy cech: {self.feature_names[:5]}...")
        
        # Zwróć odpowiedni target w zależności od trybu
        if target is not None:
            return X, target.astype(int)
        else:
            return X, None
    
    def _create_holiday_feature(self, dates: pd.Series) -> pd.Series:
        """Symulacja świąt - w rzeczywistości użyłbyś biblioteki holidays"""
        holidays = ['2024-01-01', '2024-07-04', '2024-12-25', '2024-11-28']
        holiday_dates = pd.to_datetime(holidays)
        return dates.isin(holiday_dates).astype(int)
    
    def _get_season(self, month: int) -> str:
        """Mapowanie miesiąca na porę roku"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'  
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, hyperparameter_tuning: bool = True):
        """
        🎯 Trenowanie modelu klasyfikacji
        """
        print(f"🎯 TRENOWANIE MODELU: {self.model_type.upper()}")
        print("="*50)
        
        # Ensure we have numpy arrays
        X_array = X.values if hasattr(X, 'values') else np.array(X)
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        
        # Train/test split - stratified
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array, test_size=test_size, random_state=42, stratify=y_array
        )
        
        print(f"📊 Podział danych:")
        print(f"   • Trening: {X_train.shape[0]:,} próbek")
        print(f"   • Test: {X_test.shape[0]:,} próbek")
        print(f"   • Cechy: {X_train.shape[1]}")
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning
        if hyperparameter_tuning and self.model is not None:
            print("🔍 Hyperparameter tuning...")
            self._tune_hyperparameters(X_train_scaled, y_train)
        
        # Trenowanie modelu
        print("🚂 Trenowanie modelu...")
        if self.model is not None:
            self.model.fit(X_train_scaled, y_train)
        
            # Cross-validation
            print("✅ Walidacja krzyżowa...")
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            print(f"📊 CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        print("✅ Model wytrenowany pomyślnie!")
        return self.model
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """Tuning hiperparametrów dla różnych modeli"""
        if self.model is None:
            return
            
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'xgboost':
            param_grid = {
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [50, 100, 200]
            }
        elif self.model_type == 'logistic':
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        print(f"🎯 Najlepsze parametry: {grid_search.best_params_}")
        print(f"🏆 Najlepszy score: {grid_search.best_score_:.4f}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Przewidywanie prawdopodobieństwa opóźnienia"""
        if self.model is None:
            raise ValueError("Model nie został wytrenowany!")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict_binary(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Przewidywanie binarne (0/1)"""
        if self.model is None:
            raise ValueError("Model nie został wytrenowany!")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        📊 Ewaluacja modelu - comprehensive evaluation
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany!")
            
        if X_test is None:
            X_test = self.X_test
            y_test = self.y_test
            
        print("📊 EWALUACJA MODELU KLASYFIKACJI")
        print("="*50)
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        # Basic metrics
        print("🎯 Podstawowe metryki:")
        print(classification_report(y_test, y_pred))
        
        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"🏆 ROC AUC Score: {roc_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"📊 Confusion Matrix:")
        print(cm)
        
        return {
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def _fix_feature_names(self):
        """Naprawia feature_names jeśli lista jest pusta"""
        if self.model is not None and (not hasattr(self, 'feature_names') or len(self.feature_names) == 0):
            if hasattr(self.model, 'feature_importances_'):
                n_features = len(self.model.feature_importances_)
            elif hasattr(self.model, 'coef_'):
                n_features = len(self.model.coef_.flatten())
            else:
                n_features = 0
            
            if n_features > 0:
                self.feature_names = [f'cecha_{i+1}' for i in range(n_features)]
                print(f"🔧 Naprawiono feature_names - wygenerowano {n_features} nazw cech")
    
    def get_feature_names(self):
        """Zwraca nazwy cech, generując je automatycznie jeśli potrzeba"""
        if not hasattr(self, 'feature_names') or len(self.feature_names) == 0:
            self._fix_feature_names()
        return getattr(self, 'feature_names', [])


class DelayRegressor:
    """
    📈 Regressor opóźnień lotów - przewiduje liczbę minut opóźnienia
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Inicjalizacja regressora
        
        Args:
            model_type (str): Typ modelu - 'linear', 'random_forest', 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Inicjalizacja modelu
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'xgboost':
            self.model = xgb.XGBRegressor(random_state=42)
        else:
            raise ValueError("Dostępne modele: 'linear', 'random_forest', 'xgboost'")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        🔧 Feature Engineering dla regresji - tylko opóźnione loty
        """
        print("🔧 FEATURE ENGINEERING - REGRESJA")
        print("="*50)
        
        # Filtruj tylko opóźnione loty (>0 min)
        df_delayed = df[df['delay_minutes'] > 0].copy()
        print(f"📊 Loty opóźnione: {len(df_delayed):,} z {len(df):,} ({len(df_delayed)/len(df)*100:.1f}%)")
        
        # Użyj tej samej logiki feature engineering co w klasyfikatorze
        classifier = DelayClassifier()
        X, _ = classifier.prepare_features(df_delayed)
        y = df_delayed['delay_minutes']
        
        self.feature_names = list(X.columns)
        print(f"✅ Feature engineering zakończony! Utworzono {len(self.feature_names)} cech")
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, hyperparameter_tuning: bool = True):
        """
        📈 Trenowanie modelu regresji
        """
        print(f"📈 TRENOWANIE MODELU REGRESJI: {self.model_type.upper()}")
        print("="*50)
        
        # Ensure we have numpy arrays
        X_array = X.values if hasattr(X, 'values') else np.array(X)
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array, test_size=test_size, random_state=42
        )
        
        print(f"📊 Podział danych:")
        print(f"   • Trening: {X_train.shape[0]:,} próbek")
        print(f"   • Test: {X_test.shape[0]:,} próbek")
        print(f"   • Średnie opóźnienie (trening): {y_train.mean():.1f} min")
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning
        if hyperparameter_tuning and self.model is not None:
            print("🔍 Hyperparameter tuning...")
            self._tune_hyperparameters(X_train_scaled, y_train)
        
        # Trenowanie
        print("🚂 Trenowanie modelu...")
        if self.model is not None:
            self.model.fit(X_train_scaled, y_train)
        
            # Cross-validation
            print("✅ Walidacja krzyżowa...")
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
            print(f"📊 CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store test data
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        print("✅ Model wytrenowany pomyślnie!")
        return self.model
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """Tuning hiperparametrów dla regresji"""
        if self.model is None:
            return
            
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'xgboost':
            param_grid = {
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [50, 100, 200]
            }
        elif self.model_type == 'linear':
            # Linear regression nie ma hiperparametrów do tuningu
            return
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=3, scoring='r2', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        print(f"🎯 Najlepsze parametry: {grid_search.best_params_}")
        print(f"🏆 Najlepszy score: {grid_search.best_score_:.4f}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Przewidywanie minut opóźnienia"""
        if self.model is None:
            raise ValueError("Model nie został wytrenowany!")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        📊 Ewaluacja modelu regresji
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany!")
            
        if X_test is None:
            X_test = self.X_test
            y_test = self.y_test
            
        print("📊 EWALUACJA MODELU REGRESJI")
        print("="*50)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"📊 Metryki regresji:")
        print(f"   • RMSE: {rmse:.2f} minut")
        print(f"   • MAE: {mae:.2f} minut")
        print(f"   • R²: {r2:.4f}")
        
        # Ensure y_test is numpy array for proper mean calculation
        y_test_array = y_test if isinstance(y_test, np.ndarray) else np.array(y_test)
        y_pred_array = y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred)
        
        print(f"📈 Dodatkowe statystyki:")
        print(f"   • Średnie rzeczywiste opóźnienie: {y_test_array.mean():.1f} min")
        print(f"   • Średnie przewidywane opóźnienie: {y_pred_array.mean():.1f} min")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_pred': y_pred
        }


# =============================================================================
# FUNKCJE WSPOMAGAJĄCE
# =============================================================================

def plot_model_performance(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, task_type: str = 'classification'):
    """
    📊 Wizualizacja wydajności modelu
    """
    
    if task_type == 'classification':
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'📊 Wydajność Modelu: {model_name}', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('🎯 Confusion Matrix')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = np.trapz(tpr, fpr)
        axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                      label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,1].set_title('📈 ROC Curve')
        axes[0,1].legend()
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        axes[1,0].plot(recall, precision, color='blue', lw=2)
        axes[1,0].set_title('🎯 Precision-Recall Curve')
        
        # 4. Prediction Distribution
        axes[1,1].hist(y_pred[y_true == 0], alpha=0.5, label='Nie opóźnione', bins=30)
        axes[1,1].hist(y_pred[y_true == 1], alpha=0.5, label='Opóźnione', bins=30)
        axes[1,1].set_title('📊 Rozkład przewidywań')
        axes[1,1].legend()
        
    else:  # regression
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'📈 Wydajność Modelu Regresji: {model_name}', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted
        axes[0,0].scatter(y_true, y_pred, alpha=0.5)
        axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0,0].set_title('🎯 Rzeczywiste vs Przewidywane')
        
        # 2. Residuals
        residuals = y_true - y_pred
        axes[0,1].scatter(y_pred, residuals, alpha=0.5)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_title('📊 Wykres residuów')
        
        # 3. Residuals histogram
        axes[1,0].hist(residuals, bins=30, edgecolor='black')
        axes[1,0].set_title('📈 Rozkład residuów')
        
        # 4. Error by prediction range
        axes[1,1].scatter(y_pred, np.abs(residuals), alpha=0.5)
        axes[1,1].set_title('📊 Błąd vs Przewidywanie')
    
    plt.tight_layout()
    plt.show()
    return fig


def feature_importance_plot(model, feature_names: list, top_n: int = 20):
    """
    📊 Wykres ważności cech - z automatyczną naprawą nazw cech
    """
    print("🔍 Rozpoczynam analizę ważności cech...")
    
    # Pobierz ważność cech w zależności od typu modelu
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        print("✅ Znaleziono feature_importances_")
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
        print("✅ Znaleziono coef_ - używam wartości bezwzględnych")
    else:
        print("❌ Model nie posiada informacji o ważności cech")
        return None
    
    # Konwertuj na numpy arrays
    importance = np.array(importance)
    feature_names = list(feature_names) if feature_names else []
    
    print(f"🔍 Długość feature_names: {len(feature_names)}")
    print(f"🔍 Długość importance: {len(importance)}")
    
    # AUTOMATYCZNA NAPRAWA: Jeśli feature_names jest pusty lub za krótki
    if len(feature_names) == 0 or len(feature_names) != len(importance):
        if len(feature_names) == 0:
            print("🔧 Lista feature_names jest pusta - generuję automatyczne nazwy")
        else:
            print(f"🔧 Niezgodne długości ({len(feature_names)} vs {len(importance)}) - generuję nazwy")
        
        # Stwórz automatyczne nazwy cech
        feature_names = [f'cecha_{i+1}' for i in range(len(importance))]
        print(f"✅ Wygenerowano {len(feature_names)} automatycznych nazw cech")
    
    # Teraz długości powinny być identyczne
    print(f"✅ Finalne długości - feature_names: {len(feature_names)}, importance: {len(importance)}")
    
    # Stwórz DataFrame z ważnością
    try:
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        print("✅ DataFrame utworzony pomyślnie")
    except Exception as e:
        print(f"❌ Błąd podczas tworzenia DataFrame: {e}")
        return None
    
    # Wykres top_n najważniejszych cech
    plt.figure(figsize=(12, 8))
    top_features = feature_df.head(top_n)
    
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title(f'🎯 Top {top_n} Najważniejszych Cech', fontsize=16, fontweight='bold')
    plt.xlabel('Ważność cechy')
    plt.ylabel('Cecha')
    
    plt.tight_layout()
    plt.show()
    
    return feature_df


def shap_analysis(model, X_sample: pd.DataFrame, feature_names: list, sample_size: int = 100):
    """
    🔍 Analiza SHAP - interpretabilność modelu
    """
    print("🔍 ANALIZA SHAP - INTERPRETABILNOŚĆ MODELU")
    print("="*50)
    
    try:
        # Ogranicz próbkę dla wydajności
        if len(X_sample) > sample_size:
            X_sample = X_sample.sample(n=sample_size, random_state=42)
        
        # Inicjalizuj explainer w zależności od typu modelu
        if str(type(model)).find('XGB') != -1:
            explainer = shap.Explainer(model)
        else:
            explainer = shap.Explainer(model, X_sample)
        
        shap_values = explainer(X_sample)
        
        # Summary plot
        print("📊 Tworzenie wykresu podsumowania SHAP...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title('🎯 SHAP Summary Plot - Wpływ cech na przewidywania')
        plt.tight_layout()
        plt.show()
        
        print("✅ Analiza SHAP zakończona!")
        return shap_values
        
    except Exception as e:
        print(f"❌ Błąd podczas analizy SHAP: {e}")
        return None


def predict_delay(flight_details: Dict[str, Any], classifier_model: DelayClassifier, regressor_model: DelayRegressor) -> Optional[Dict[str, Any]]:
    """
    🎯 Funkcja do przewidywania opóźnień dla nowych lotów
    """
    print("🎯 PRZEWIDYWANIE OPÓŹNIENIA LOTU")
    print("="*40)
    
    try:
        # Konwersja do DataFrame
        df_single = pd.DataFrame([flight_details])
        
        # Feature engineering w trybie przewidywania
        X_single, _ = classifier_model.prepare_features(df_single, prediction_mode=True)
        
        # Klasyfikacja - czy będzie opóźniony?
        delay_probability = classifier_model.predict(X_single)[0]
        is_delayed = delay_probability > 0.5
        
        # Regresja - o ile minut opóźnienie?
        if is_delayed:
            expected_delay = regressor_model.predict(X_single)[0]
        else:
            expected_delay = 0
        
        result = {
            'is_delayed': is_delayed,
            'delay_probability': delay_probability,
            'expected_delay_minutes': max(0, expected_delay),
            'risk_level': 'Wysoki' if delay_probability > 0.7 else 'Średni' if delay_probability > 0.3 else 'Niski'
        }
        
        print(f"✈️  Lot: {flight_details.get('airline', 'N/A')} {flight_details.get('origin', 'N/A')} → {flight_details.get('destination', 'N/A')}")
        print(f"📅 Data: {flight_details.get('flight_date', 'N/A')} o {flight_details.get('scheduled_departure', 'N/A')}")
        print(f"🎯 Prawdopodobieństwo opóźnienia: {delay_probability:.1%}")
        print(f"⏰ Przewidywane opóźnienie: {result['expected_delay_minutes']:.0f} minut")
        print(f"⚠️  Poziom ryzyka: {result['risk_level']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Błąd podczas przewidywania: {e}")
        return None


def save_models(classifier: DelayClassifier, regressor: DelayRegressor, filename_base: str = 'best_model'):
    """
    💾 Zapisywanie modeli do plików
    """
    try:
        classifier_path = f"{filename_base}_classifier.joblib"
        regressor_path = f"{filename_base}_regressor.joblib"
        
        # Zapisz modele
        joblib.dump(classifier, classifier_path)
        joblib.dump(regressor, regressor_path)
        
        print(f"✅ Modele zapisane:")
        print(f"   • Klasyfikator: {classifier_path}")
        print(f"   • Regressor: {regressor_path}")
        
        return classifier_path, regressor_path
        
    except Exception as e:
        print(f"❌ Błąd podczas zapisywania: {e}")
        return None, None


def load_models(classifier_path: str, regressor_path: str) -> Tuple[Optional[DelayClassifier], Optional[DelayRegressor]]:
    """
    📂 Wczytywanie modeli z plików
    """
    try:
        classifier = joblib.load(classifier_path)
        regressor = joblib.load(regressor_path)
        
        print(f"✅ Modele wczytane:")
        print(f"   • Klasyfikator: {classifier_path}")
        print(f"   • Regressor: {regressor_path}")
        
        return classifier, regressor
        
    except Exception as e:
        print(f"❌ Błąd podczas wczytywania: {e}")
        return None, None


# =============================================================================
# DODATKOWE NARZĘDZIA DO WIZUALIZACJI
# =============================================================================

def create_model_comparison_plot(models_results: Dict[str, Dict], metric: str = 'roc_auc'):
    """
    📊 Porównanie wydajności różnych modeli
    """
    model_names = list(models_results.keys())
    scores = [models_results[model][metric] for model in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, scores, color=['skyblue', 'lightgreen', 'orange'])
    plt.title(f'📊 Porównanie Modeli - {metric.upper()}', fontsize=14, fontweight='bold')
    plt.ylabel(f'{metric.upper()} Score')
    plt.ylim(0, 1)
    
    # Dodaj wartości na słupkach
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()


def analyze_feature_importance(classifier, top_n: int = 20):
    """
    🎯 Kompleksowa analiza ważności cech - automatycznie naprawia nazwy
    """
    if classifier.model is None:
        print("❌ Model nie został wytrenowany!")
        return None
    
    # Automatycznie napraw feature_names jeśli potrzeba
    classifier._fix_feature_names()
    
    # Użyj naprawionych nazw cech
    feature_names = classifier.get_feature_names()
    
    if len(feature_names) == 0:
        print("❌ Nie można uzyskać nazw cech")
        return None
    
    print(f"✅ Analizuję ważność {len(feature_names)} cech dla modelu {classifier.model_type}")
    
    # Wywołaj standardową funkcję z naprawionymi nazwami
    return feature_importance_plot(
        model=classifier.model,
        feature_names=feature_names,
        top_n=top_n
    )


if __name__ == "__main__":
    print("🤖 MODUŁ MACHINE LEARNING - AIRLINE ANALYTICS")
    print("="*50)
    print("✅ Wszystkie klasy i funkcje załadowane pomyślnie!")
    print("\n📖 Dostępne klasy:")
    print("   • DelayClassifier - klasyfikacja opóźnień")
    print("   • DelayRegressor - regresja opóźnień")
    print("\n🔧 Dostępne funkcje:")
    print("   • plot_model_performance() - wizualizacja wydajności")
    print("   • feature_importance_plot() - ważność cech")
    print("   • shap_analysis() - interpretabilność")
    print("   • predict_delay() - przewidywanie dla nowych lotów")
    print("   • create_model_comparison_plot() - porównanie modeli") 