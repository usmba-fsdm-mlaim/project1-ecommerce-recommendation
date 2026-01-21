import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import json

# Ajoute le dossier parent au chemin pour pouvoir importer le script principal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing import DataPreprocessor

# --- Fixture (Données de test) ---
@pytest.fixture
def sample_csv_path():
    """
    Crée un fichier CSV temporaire avec des données factices 
    qui imitent la structure réelle des données Amazon.
    """
    data = {
        'id': ['prod1', 'prod1', 'prod2', 'prod3'],  # prod1 est dupliqué
        'reviews.username': ['UserA', 'UserA', 'UserB', 'UserC'],
        'reviews.date': ['2023-01-01', '2023-01-01', '2023-02-01', '2023-03-01'],
        'brand': ['BrandA', 'BrandA', 'BrandB', 'BrandC'],
        'name': ['Product A', 'Product A', ' Product B ', 'Product C'], # Espace à nettoyer
        'reviews.text': ['Great product', 'Great product', 'Bad quality', None], # Un texte manquant
        'reviews.rating': [5.0, 5.0, 1.0, 4.0],
        'prices': [
            '[{"amountMin": 100.0}]', 
            '[{"amountMin": 100.0}]', 
            '[{"amountMin": 50.0}]', 
            None
        ],
        'categories': ['Electronics,Computers', 'Electronics,Computers', 'Home', 'Books'],
        'dateAdded': ['2022-01-01', '2022-01-01', '2022-01-01', '2022-01-01'],
        'reviews.numHelpful': [10, 10, 0, 2],
        'reviews.title': ['Title1', 'Title1', 'Title2', 'Title3'],
        'asins': ['A1', 'A1', 'A2', 'A3']
    }
    
    df = pd.DataFrame(data)
    
    # Création du fichier physique temporaire
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        df.to_csv(tmp.name, index=False)
        temp_path = tmp.name
        
    yield temp_path  # Passe le chemin au test
    
    # Nettoyage après le test
    if os.path.exists(temp_path):
        os.remove(temp_path)

# --- Tests Unitaires ---

def test_load_data(sample_csv_path):
    processor = DataPreprocessor(sample_csv_path)
    processor.load_data()
    assert processor.df is not None
    assert len(processor.df) == 4

def test_clean_basic_fields(sample_csv_path):
    """Vérifie la suppression des doublons et le nettoyage des espaces"""
    processor = DataPreprocessor(sample_csv_path)
    processor.load_data()
    processor.clean_basic_fields()
    
    # On attend : 
    # - Suppression du doublon exact (ligne 2)
    # - Suppression de la ligne avec review manquante (ligne 4)
    # Reste : 2 lignes (prod1 et prod2)
    assert len(processor.df) == 2
    
    # Vérifie le nettoyage des espaces (' Product B ' -> 'Product B')
    prod2_name = processor.df[processor.df['id'] == 'prod2']['name'].iloc[0]
    assert prod2_name == 'Product B'

def test_parse_prices(sample_csv_path):
    """Vérifie l'extraction du prix depuis le JSON"""
    processor = DataPreprocessor(sample_csv_path)
    processor.load_data()
    processor.parse_prices()
    
    # Le prix de prod1 doit être 100.0
    price = processor.df[processor.df['id'] == 'prod1']['price'].iloc[0]
    assert price == 100.0

def test_parse_categories(sample_csv_path):
    """Vérifie que seule la catégorie principale est gardée"""
    processor = DataPreprocessor(sample_csv_path)
    processor.load_data()
    processor.parse_categories()
    
    # 'Electronics,Computers' -> 'Electronics'
    category = processor.df[processor.df['id'] == 'prod1']['main_category'].iloc[0]
    assert category == 'Electronics'

def test_create_user_features(sample_csv_path):
    """Vérifie la création des statistiques utilisateurs"""
    processor = DataPreprocessor(sample_csv_path)
    processor.load_data()
    processor.clean_basic_fields() # Nécessaire pour nettoyer avant
    processor.clean_user_data()    # Nécessaire pour créer user_id
    processor.create_user_features()
    
    assert 'user_avg_rating' in processor.df.columns
    assert 'user_review_count' in processor.df.columns

def test_end_to_end_pipeline(sample_csv_path):
    """Teste tout le pipeline d'un coup"""
    processor = DataPreprocessor(sample_csv_path)
    
    with tempfile.NamedTemporaryFile(suffix='.csv') as tmp_out:
        final_df = processor.run_pipeline(output_path=tmp_out.name)
        
        # Vérifications finales
        assert not final_df.empty
        assert 'product_id' in final_df.columns  # Renommé de 'id'
        assert 'user_id' in final_df.columns
        assert 'rating' in final_df.columns
        
        # Vérifier qu'il n'y a pas de valeurs nulles
        assert final_df['rating'].isnull().sum() == 0