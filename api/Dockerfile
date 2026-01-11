# Utiliser une image Python légère
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source
COPY . .

# Exposer le port
EXPOSE 8000

# Lancer l'application (app:app au lieu de main:app)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]