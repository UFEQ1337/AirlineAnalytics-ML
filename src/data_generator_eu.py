"""
Generator danych lotniczych dla rynku europejskiego/polskiego.
Generuje realistyczne dane lotów z europejskimi liniami lotniczymi i lotniskami.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random

fake = Faker('pl_PL')  # Polskie lokalizacje

class EuropeanFlightDataGenerator:
    def __init__(self, n_records=50000):
        self.n_records = n_records
        
        # Europejskie linie lotnicze (fokus na Polsce)
        self.airlines = [
            'LOT Polish Airlines', 'Ryanair', 'Wizz Air', 'Lufthansa', 
            'KLM', 'Air France', 'British Airways', 'easyJet',
            'Eurowings', 'Austrian Airlines', 'SAS', 'Norwegian',
            'Alitalia', 'Iberia', 'TAP Air Portugal', 'Swiss'
        ]
        
        # Europejskie lotniska (fokus na Polsce i sąsiadów)
        self.airports = {
            # Polska
            'WAW': 'Warszawa Chopin', 'KRK': 'Kraków Balice', 'GDN': 'Gdańsk Rębiechowo',
            'WRO': 'Wrocław Strachowice', 'KTW': 'Katowice Pyrzowice', 'POZ': 'Poznań Ławica',
            'RZE': 'Rzeszów Jasionka', 'LUZ': 'Lublin',
            
            # Niemcy
            'FRA': 'Frankfurt', 'MUC': 'Monachium', 'DUS': 'Düsseldorf', 'BER': 'Berlin',
            'HAM': 'Hamburg', 'CGN': 'Kolonia',
            
            # Europa Zachodnia
            'LHR': 'Londyn Heathrow', 'CDG': 'Paryż Charles de Gaulle', 'AMS': 'Amsterdam',
            'MAD': 'Madryt', 'FCO': 'Rzym', 'VIE': 'Wiedeń', 'ZUR': 'Zurych',
            
            # Europa Środkowa/Wschodnia
            'PRG': 'Praga', 'BUD': 'Budapeszt', 'VNO': 'Wilno', 'RIX': 'Ryga',
            'TLL': 'Tallin', 'KEF': 'Reykjavik', 'ARN': 'Sztokholm'
        }
        
        # Typy samolotów używane w Europie
        self.aircraft_types = [
            'Boeing 737-800', 'Boeing 737-900', 'Airbus A320', 'Airbus A321',
            'Airbus A319', 'Boeing 787-8', 'Airbus A330-300', 'Embraer 175',
            'ATR 72', 'Bombardier Q400', 'Airbus A350-900', 'Boeing 777-300ER'
        ]
        
        # Przyczyny opóźnień (europejskie specyfiki)
        self.delay_reasons = ['None', 'Weather', 'Air Traffic', 'Mechanical', 'Security', 'Strike', 'Airport Issues']
        
        # Dystanse między europejskimi miastami (w kilometrach)
        self.city_distances = {
            ('WAW', 'FRA'): 1160, ('WAW', 'LHR'): 1460, ('WAW', 'CDG'): 1370,
            ('KRK', 'VIE'): 290, ('KRK', 'MUC'): 430, ('GDN', 'ARN'): 340,
            ('WAW', 'PRG'): 520, ('WAW', 'BUD'): 550, ('WAW', 'VNO'): 430,
            ('WAW', 'AMS'): 1050, ('WAW', 'ZUR'): 1100, ('KRK', 'FCO'): 1100,
            ('WRO', 'BER'): 280, ('POZ', 'HAM'): 460, ('KTW', 'DUS'): 720
        }
    
    def get_distance(self, origin, destination):
        """Zwraca dystans między europejskimi miastami"""
        if (origin, destination) in self.city_distances:
            return self.city_distances[(origin, destination)]
        elif (destination, origin) in self.city_distances:
            return self.city_distances[(destination, origin)]
        else:
            # Generuj realistyczny dystans dla Europy (krótsze niż USA)
            return random.randint(150, 2500)
    
    def get_european_delay_probability(self, hour, weekday, month, distance, origin, destination):
        """Oblicza prawdopodobieństwo opóźnienia dla ruchu europejskiego"""
        base_prob = 0.25  # nieco mniejsze niż USA
        
        # Rush hours europejskie (6-8, 18-20)
        if hour in [6, 7, 8, 18, 19, 20]:
            base_prob += 0.12
        elif hour in [5, 21, 22]:
            base_prob -= 0.08
        
        # Weekend - mniej lotów biznesowych
        if weekday in [5, 6]:
            base_prob -= 0.06
        elif weekday in [0, 4]:  # poniedziałek/piątek
            base_prob += 0.04
        
        # Europejskie wzorce sezonowe
        if month in [12, 1, 2]:  # zima - częste problemy pogodowe
            base_prob += 0.15
        elif month in [7, 8]:  # lato - season szczytowy
            base_prob += 0.08
        elif month in [3, 4, 9, 10]:  # spring/autumn - najlepsze warunki
            base_prob -= 0.05
        
        # Specyficzne lotniska z problemami
        problematic_airports = ['LHR', 'CDG', 'FRA', 'AMS']  # duże huby
        if origin in problematic_airports or destination in problematic_airports:
            base_prob += 0.08
        
        # Polskie lotniska - generalnie lepsze statystyki
        polish_airports = ['WAW', 'KRK', 'GDN', 'WRO', 'KTW', 'POZ']
        if origin in polish_airports and destination in polish_airports:
            base_prob -= 0.05
        
        # Krótkie loty europejskie - mniej problemów
        if distance < 800:
            base_prob -= 0.04
        elif distance > 2000:
            base_prob += 0.06
        
        return max(0.03, min(0.65, base_prob))
    
    def generate_european_delay(self, prob):
        """Generuje opóźnienie charakterystyczne dla Europy"""
        if random.random() > prob:
            return 0
        
        # Europejski rozkład opóźnień (zwykle krótsze niż USA)
        if random.random() < 0.75:
            return random.randint(5, 25)  # krótkie opóźnienia
        elif random.random() < 0.92:
            return random.randint(26, 75)  # średnie opóźnienia
        else:
            return random.randint(76, 240)  # długie opóźnienia
    
    def get_european_delay_reason(self, delay_minutes):
        """Określa przyczynę opóźnienia w kontekście europejskim"""
        if delay_minutes == 0:
            return 'None'
        
        prob = random.random()
        if prob < 0.35:
            return 'Weather'
        elif prob < 0.60:
            return 'Air Traffic'
        elif prob < 0.80:
            return 'Mechanical'
        elif prob < 0.90:
            return 'Strike'  # częstsze w Europie
        elif prob < 0.95:
            return 'Airport Issues'
        else:
            return 'Security'
    
    def generate_european_flights_data(self):
        """Generuje dataset europejskich lotów"""
        flights_data = []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        print(f"Generowanie {self.n_records} europejskich rekordów lotów...")
        
        airport_codes = list(self.airports.keys())
        
        for i in range(self.n_records):
            if i % 10000 == 0:
                print(f"Wygenerowano {i} rekordów...")
            
            flight_date = fake.date_between(start_date=start_date, end_date=end_date)
            
            # Europejskie godziny lotów (6:00-23:00, mniej red-eye)
            scheduled_hour = random.randint(6, 23)
            scheduled_minute = random.choice([0, 10, 15, 25, 30, 40, 45, 55])
            scheduled_departure = f"{scheduled_hour:02d}:{scheduled_minute:02d}"
            
            # Lotniska z większym prawdopodobieństwem dla polskich
            if random.random() < 0.4:  # 40% lotów z/do Polski
                polish_airports = ['WAW', 'KRK', 'GDN', 'WRO', 'KTW', 'POZ']
                origin = random.choice(polish_airports)
                destination = random.choice([a for a in airport_codes if a != origin])
            else:
                origin = random.choice(airport_codes)
                destination = random.choice([a for a in airport_codes if a != origin])
            
            distance = self.get_distance(origin, destination)
            
            delay_prob = self.get_european_delay_probability(
                scheduled_hour, flight_date.weekday(), flight_date.month, 
                distance, origin, destination
            )
            
            delay_minutes = self.generate_european_delay(delay_prob)
            
            # Rzeczywista godzina odlotu
            scheduled_dt = datetime.combine(flight_date, datetime.strptime(scheduled_departure, "%H:%M").time())
            actual_dt = scheduled_dt + timedelta(minutes=delay_minutes)
            actual_departure = actual_dt.strftime("%H:%M")
            
            delay_reason = self.get_european_delay_reason(delay_minutes)
            
            # Wybór linii lotniczej na podstawie trasy
            if origin in ['WAW', 'KRK', 'GDN', 'WRO', 'KTW', 'POZ'] or destination in ['WAW', 'KRK', 'GDN', 'WRO', 'KTW', 'POZ']:
                if random.random() < 0.3:
                    airline = 'LOT Polish Airlines'
                elif random.random() < 0.6:
                    airline = random.choice(['Ryanair', 'Wizz Air'])
                else:
                    airline = random.choice(self.airlines)
            else:
                airline = random.choice(self.airlines)
            
            flight_record = {
                'flight_date': flight_date,
                'airline': airline,
                'origin': origin,
                'origin_city': self.airports[origin],
                'destination': destination,
                'destination_city': self.airports[destination],
                'scheduled_departure': scheduled_departure,
                'actual_departure': actual_departure,
                'delay_minutes': delay_minutes,
                'delay_reason': delay_reason,
                'distance_km': distance,
                'aircraft_type': random.choice(self.aircraft_types),
                'country_origin': self.get_country(origin),
                'country_destination': self.get_country(destination)
            }
            
            flights_data.append(flight_record)
        
        print(f"Generowanie zakończone! Utworzono {len(flights_data)} europejskich rekordów.")
        return pd.DataFrame(flights_data)
    
    def get_country(self, airport_code):
        """Zwraca kraj dla kodu lotniska"""
        country_mapping = {
            'WAW': 'Polska', 'KRK': 'Polska', 'GDN': 'Polska', 'WRO': 'Polska',
            'KTW': 'Polska', 'POZ': 'Polska', 'RZE': 'Polska', 'LUZ': 'Polska',
            'FRA': 'Niemcy', 'MUC': 'Niemcy', 'DUS': 'Niemcy', 'BER': 'Niemcy',
            'HAM': 'Niemcy', 'CGN': 'Niemcy',
            'LHR': 'Wielka Brytania', 'CDG': 'Francja', 'AMS': 'Holandia',
            'MAD': 'Hiszpania', 'FCO': 'Włochy', 'VIE': 'Austria', 'ZUR': 'Szwajcaria',
            'PRG': 'Czechy', 'BUD': 'Węgry', 'VNO': 'Litwa', 'RIX': 'Łotwa',
            'TLL': 'Estonia', 'KEF': 'Islandia', 'ARN': 'Szwecja'
        }
        return country_mapping.get(airport_code, 'Inne')

def generate_european_data(filename="data/raw/european_flights_data.csv", n_records=50000):
    """Główna funkcja do generowania europejskich danych"""
    generator = EuropeanFlightDataGenerator(n_records)
    df = generator.generate_european_flights_data()
    
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df.to_csv(filename, index=False)
    print(f"Europejskie dane zapisane do pliku: {filename}")
    
    return df

if __name__ == "__main__":
    # Generuj europejskie dane
    df = generate_european_data()
    print(f"Wygenerowano {len(df)} europejskich rekordów lotów")
    print(f"Polskie loty: {len(df[df['country_origin'] == 'Polska'])} jako origin")
    print(f"Top 5 tras:")
    routes = df['origin'].astype(str) + ' → ' + df['destination'].astype(str)
    print(routes.value_counts().head()) 