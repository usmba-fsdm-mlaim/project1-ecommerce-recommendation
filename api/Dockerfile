# 1. On part d'une version légère de Python
FROM python:3.9-slim

# 2. On crée le dossier de travail dans le conteneur
WORKDIR /app

# 3. On copie d'abord les requirements (astuce pour que le build soit plus rapide)
COPY requirements.txt .

# 4. On installe les librairies
RUN pip install --no-cache-dir -r requirements.txt

# 5. On copie tout le reste du code
COPY . .

# 6. On indique que l'app tourne sur le port 8000
EXPOSE 8000

# 7. La commande de démarrage quand le conteneur se lance
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]