"""
Generator danych lotniczych z realistycznymi wzorcami opóźnień.
Generuje 50,000 rekordów lotów z ostatnich 2 lat.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random

fake = Faker()

class FlightDataGenerator:
    def __init__(self, n_records=50000):
        self.n_records = n_records
        
        # Główne linie lotnicze
        self.airlines = [
            'American Airlines', 'Delta Air Lines', 'United Airlines', 
            'Southwest Airlines', 'JetBlue Airways', 'Alaska Airlines',
            'Spirit Airlines', 'Frontier Airlines', 'Allegiant Air', 'Hawaiian Airlines'
        ]
        
        # Główne porty lotnicze USA
        self.airports = [
            'JFK', 'LAX', 'ORD', 'DFW', 'DEN', 'SFO', 'SEA', 'LAS', 'PHX', 'IAH',
            'CLT', 'MIA', 'BOS', 'MSP', 'FLL', 'DTW', 'PHL', 'LGA', 'BWI', 'MDW'
        ]
        
        # Typy samolotów
        self.aircraft_types = [
            'Boeing 737-800', 'Boeing 737-900', 'Airbus A320', 'Airbus A321',
            'Boeing 757-200', 'Boeing 767-300', 'Boeing 777-200', 'Boeing 787-8',
            'Airbus A330-200', 'Embraer 175', 'CRJ-900', 'ATR 72'
        ]
        
        # Przyczyny opóźnień
        self.delay_reasons = ['None', 'Weather', 'Air Traffic', 'Mechanical', 'Security']
        
        # Dystanse między miastami (w milach)
        self.city_distances = {
            ('JFK', 'LAX'): 2475, ('ORD', 'DFW'): 925, ('DEN', 'SFO'): 967,
            ('SEA', 'LAS'): 867, ('PHX', 'IAH'): 1009, ('CLT', 'MIA'): 647,
            ('BOS', 'MSP'): 1123, ('FLL', 'DTW'): 1056, ('PHL', 'LGA'): 95,
            ('BWI', 'MDW'): 621
        }
    
    def get_distance(self, origin, destination):
        """Zwraca dystans między miastami lub generuje realistyczny dystans"""
        if (origin, destination) in self.city_distances:
            return self.city_distances[(origin, destination)]
        elif (destination, origin) in self.city_distances:
            return self.city_distances[(destination, origin)]
        else:
            # Generuj realistyczny dystans bazujący na średnich dystansach krajowych
            return random.randint(200, 3000)
    
    def get_delay_probability(self, hour, weekday, month, distance):
        """Oblicza prawdopodobieństwo opóźnienia na podstawie różnych czynników"""
        base_prob = 0.3  # podstawowe prawdopodobieństwo opóźnienia
        
        # Rush hours (7-9, 17-19) - większe prawdopodobieństwo
        if hour in [7, 8, 17, 18, 19]:
            base_prob += 0.15
        # Wczesne loty - mniejsze prawdopodobieństwo
        elif hour in [5, 6]:
            base_prob -= 0.1
        
        # Weekend - mniejsze prawdopodobieństwo
        if weekday in [5, 6]:  # sobota, niedziela
            base_prob -= 0.05
        # Poniedziałek/piątek - większe prawdopodobieństwo
        elif weekday in [0, 4]:
            base_prob += 0.05
        
        # Miesiące zimowe - większe prawdopodobieństwo
        if month in [12, 1, 2]:
            base_prob += 0.1
        # Miesiące letnie - mniejsze prawdopodobieństwo
        elif month in [6, 7, 8]:
            base_prob -= 0.05
        
        # Długie loty - większe prawdopodobieństwo opóźnień
        if distance > 2000:
            base_prob += 0.08
        elif distance < 500:
            base_prob -= 0.03
        
        return max(0.05, min(0.7, base_prob))  # ograniczenie do 5-70%
    
    def generate_delay(self, prob):
        """Generuje opóźnienie w minutach"""
        if random.random() > prob:
            return 0  # brak opóźnienia
        
        # Rozkład opóźnień - większość krótkich, nieliczne bardzo długie
        if random.random() < 0.7:
            return random.randint(5, 30)  # krótkie opóźnienia
        elif random.random() < 0.9:
            return random.randint(31, 90)  # średnie opóźnienia
        else:
            return random.randint(91, 300)  # długie opóźnienia
    
    def get_delay_reason(self, delay_minutes):
        """Określa przyczynę opóźnienia"""
        if delay_minutes == 0:
            return 'None'
        
        # Rozkład przyczyn opóźnień
        prob = random.random()
        if prob < 0.50:
            return 'Weather'
        elif prob < 0.75:
            return 'Air Traffic'
        elif prob < 0.90:
            return 'Mechanical'
        else:
            return 'Security'
    
    def generate_flights_data(self):
        """Generuje kompletny dataset lotów"""
        flights_data = []
        
        # Generuj daty z ostatnich 2 lat
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        print(f"Generowanie {self.n_records} rekordów lotów...")
        
        for i in range(self.n_records):
            if i % 10000 == 0:
                print(f"Wygenerowano {i} rekordów...")
            
            # Losowa data lotu
            flight_date = fake.date_between(start_date=start_date, end_date=end_date)
            
            # Losowa godzina odlotu (5:00-23:00)
            scheduled_hour = random.randint(5, 23)
            scheduled_minute = random.choice([0, 15, 30, 45])
            scheduled_departure = f"{scheduled_hour:02d}:{scheduled_minute:02d}"
            
            # Lotniska
            origin = random.choice(self.airports)
            destination = random.choice([a for a in self.airports if a != origin])
            
            # Dystans
            distance = self.get_distance(origin, destination)
            
            # Prawdopodobieństwo opóźnienia
            delay_prob = self.get_delay_probability(
                scheduled_hour, 
                flight_date.weekday(), 
                flight_date.month,
                distance
            )
            
            # Generuj opóźnienie
            delay_minutes = self.generate_delay(delay_prob)
            
            # Oblicz rzeczywistą godzinę odlotu
            scheduled_dt = datetime.combine(flight_date, datetime.strptime(scheduled_departure, "%H:%M").time())
            actual_dt = scheduled_dt + timedelta(minutes=delay_minutes)
            actual_departure = actual_dt.strftime("%H:%M")
            
            # Przyczyna opóźnienia
            delay_reason = self.get_delay_reason(delay_minutes)
            
            flight_record = {
                'flight_date': flight_date,
                'airline': random.choice(self.airlines),
                'origin': origin,
                'destination': destination,
                'scheduled_departure': scheduled_departure,
                'actual_departure': actual_departure,
                'delay_minutes': delay_minutes,
                'delay_reason': delay_reason,
                'distance_miles': distance,
                'aircraft_type': random.choice(self.aircraft_types)
            }
            
            flights_data.append(flight_record)
        
        print(f"Generowanie zakończone! Utworzono {len(flights_data)} rekordów.")
        return pd.DataFrame(flights_data)

def generate_and_save_data(filename="data/raw/flights_data.csv", n_records=50000):
    """Główna funkcja do generowania i zapisywania danych"""
    generator = FlightDataGenerator(n_records)
    df = generator.generate_flights_data()
    
    # Sprawdź czy katalog istnieje i utwórz go jeśli nie
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Zapisz dane
    df.to_csv(filename, index=False)
    print(f"Dane zapisane do pliku: {filename}")
    
    return df

if __name__ == "__main__":
    # Generuj dane jeśli skrypt uruchomiony bezpośrednio
    # Użyj poprawnej ścieżki względnej gdy uruchamiany z katalogu src
    df = generate_and_save_data("../data/raw/flights_data.csv")
    print("\nPodgląd pierwszych 5 wierszy:")
    print(df.head())
    print(f"\nRozmiar datasetu: {df.shape}")
    print(f"\nPodstawowe statystyki opóźnień:")
    print(df['delay_minutes'].describe()) 