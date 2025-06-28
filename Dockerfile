# ✈️ AIRLINE ANALYTICS ML - DOCKERFILE
# ===================================
# 
# Containeryzacja aplikacji dashboard Streamlit
# Obraz produkcyjny z optymalizacjami

FROM python:3.9-slim

# Metadane
LABEL maintainer="AirlineAnalytics-ML Team"
LABEL description="Airline Analytics ML Dashboard"
LABEL version="1.0.0"

# Zmienne środowiskowe
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Utwórz użytkownika aplikacji (security best practice)
RUN useradd --create-home --shell /bin/bash app

# Zainstaluj zależności systemowe
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Utwórz katalog aplikacji
WORKDIR /app

# Skopiuj requirements i zainstaluj zależności Python
COPY requirements_prod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_prod.txt

# Skopiuj kod aplikacji
COPY . .

# Stwórz katalogi dla danych i wyników
RUN mkdir -p data/processed data/raw results logs && \
    chown -R app:app /app

# Przełącz na użytkownika aplikacji
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Eksponuj port Streamlit
EXPOSE 8501

# Domyślna komenda uruchomienia
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 