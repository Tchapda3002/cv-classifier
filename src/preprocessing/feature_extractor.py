"""
src/preprocessing/feature_extractor.py
Module pour l'extraction de features à partir du texte des CV
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
from pathlib import Path


class CVFeatureExtractor:
    """Classe pour extraire des features numériques à partir de texte de CV"""
    
    def __init__(self, method='tfidf', max_features=5000, ngram_range=(1, 2)):
        """
        Initialiser l'extracteur de features
        
        Args:
            method: 'tfidf' ou 'count' pour le type de vectorisation
            max_features: Nombre maximum de features à extraire
            ngram_range: Tuple (min_n, max_n) pour les n-grams
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.feature_names = None
        
    def create_vectorizer(self):
        """Créer le vectorizer selon la méthode choisie"""
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=2,  # Mot doit apparaître dans au moins 2 documents
                max_df=0.95,  # Mot ne doit pas apparaître dans plus de 95% des documents
                sublinear_tf=True,  # Utiliser le scaling sublinéaire pour tf
                use_idf=True,
                smooth_idf=True
            )
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=2,
                max_df=0.95
            )
        else:
            raise ValueError(f"Méthode '{self.method}' non reconnue. Utilisez 'tfidf' ou 'count'")
        
        print(f" Vectorizer {self.method.upper()} créé")
        print(f"   - Max features: {self.max_features}")
        print(f"   - N-gram range: {self.ngram_range}")
        
    def fit_transform(self, texts):
        """
        Entraîner le vectorizer et transformer les textes
        
        Args:
            texts: Liste ou Series de textes
        
        Returns:
            Matrice sparse de features
        """
        if self.vectorizer is None:
            self.create_vectorizer()
        
        print(f"\n Entraînement du vectorizer sur {len(texts)} documents...")
        X = self.vectorizer.fit_transform(texts)
        
        # Sauvegarder les noms de features
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f" Transformation terminée!")
        print(f"   - Shape de la matrice: {X.shape}")
        print(f"   - Nombre de features: {X.shape[1]}")
        print(f"   - Sparsité: {(1.0 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}%")
        
        return X
    
    def transform(self, texts):
        """
        Transformer de nouveaux textes (vectorizer déjà entraîné)
        
        Args:
            texts: Liste ou Series de textes
        
        Returns:
            Matrice sparse de features
        """
        if self.vectorizer is None:
            raise ValueError("Le vectorizer n'est pas entraîné. Utilisez fit_transform() d'abord.")
        
        print(f"\n Transformation de {len(texts)} nouveaux documents...")
        X = self.vectorizer.transform(texts)
        print(f" Transformation terminée! Shape: {X.shape}")
        
        return X
    
    def get_top_features(self, n=20):
        """
        Obtenir les top N features par importance
        
        Args:
            n: Nombre de features à retourner
        
        Returns:
            Liste des top features
        """
        if self.feature_names is None:
            raise ValueError("Aucune feature disponible. Entraînez le vectorizer d'abord.")
        
        # Pour TF-IDF, on peut calculer l'importance moyenne
        return list(self.feature_names[:n])
    
    def save(self, filepath='models_saved/tfidf_vectorizer.pkl'):
        """
        Sauvegarder le vectorizer
        
        Args:
            filepath: Chemin où sauvegarder le vectorizer
        """
        if self.vectorizer is None:
            print(" Aucun vectorizer à sauvegarder")
            return
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f" Vectorizer sauvegardé dans {filepath}")
    
    def load(self, filepath='models_saved/tfidf_vectorizer.pkl'):
        """
        Charger un vectorizer sauvegardé
        
        Args:
            filepath: Chemin du vectorizer à charger
        """
        try:
            with open(filepath, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            self.feature_names = self.vectorizer.get_feature_names_out()
            print(f" Vectorizer chargé depuis {filepath}")
            print(f"   - Nombre de features: {len(self.feature_names)}")
            
        except FileNotFoundError:
            print(f" Fichier {filepath} non trouvé")
    
    def reduce_dimensions(self, X, n_components=100):
        """
        Réduire la dimensionnalité avec SVD
        
        Args:
            X: Matrice de features
            n_components: Nombre de composantes à garder
        
        Returns:
            Matrice réduite
        """
        print(f"\n Réduction de dimensionnalité: {X.shape[1]} → {n_components} dimensions...")
        
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_reduced = svd.fit_transform(X)
        
        explained_variance = svd.explained_variance_ratio_.sum() * 100
        
        print(f" Réduction terminée!")
        print(f"   - Nouvelle shape: {X_reduced.shape}")
        print(f"   - Variance expliquée: {explained_variance:.2f}%")
        
        return X_reduced, svd


class Word2VecExtractor:
    """Extracteur de features basé sur Word2Vec"""
    
    def __init__(self, vector_size=100, window=5, min_count=2):
        """
        Initialiser l'extracteur Word2Vec
        
        Args:
            vector_size: Dimension des vecteurs de mots
            window: Taille de la fenêtre contextuelle
            min_count: Nombre minimum d'occurrences d'un mot
        """
        try:
            from gensim.models import Word2Vec
            self.Word2Vec = Word2Vec
        except ImportError:
            print(" gensim non installé. Installez avec: pip install gensim")
            return
        
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
        
    def train(self, texts, epochs=10):
        """
        Entraîner le modèle Word2Vec
        
        Args:
            texts: Liste de textes (strings)
            epochs: Nombre d'epochs d'entraînement
        
        Returns:
            Modèle Word2Vec entraîné
        """
        # Tokeniser les textes
        sentences = [text.split() for text in texts]
        
        print(f"\n Entraînement Word2Vec sur {len(sentences)} documents...")
        print(f"   - Vector size: {self.vector_size}")
        print(f"   - Window: {self.window}")
        print(f"   - Epochs: {epochs}")
        
        self.model = self.Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            epochs=epochs,
            seed=42
        )
        
        print(f" Entraînement terminé!")
        print(f"   - Vocabulaire: {len(self.model.wv)} mots")
        
        return self.model
    
    def text_to_vector(self, text):
        """
        Convertir un texte en vecteur (moyenne des word vectors)
        
        Args:
            text: Texte à vectoriser
        
        Returns:
            Vecteur numpy
        """
        if self.model is None:
            raise ValueError("Modèle non entraîné")
        
        words = text.split()
        word_vectors = []
        
        for word in words:
            if word in self.model.wv:
                word_vectors.append(self.model.wv[word])
        
        if len(word_vectors) == 0:
            return np.zeros(self.vector_size)
        
        return np.mean(word_vectors, axis=0)
    
    def transform(self, texts):
        """
        Transformer une liste de textes en vecteurs
        
        Args:
            texts: Liste de textes
        
        Returns:
            Matrice numpy de vecteurs
        """
        print(f"\n Transformation de {len(texts)} textes en vecteurs...")
        vectors = np.array([self.text_to_vector(text) for text in texts])
        print(f" Transformation terminée! Shape: {vectors.shape}")
        return vectors


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def extract_features_pipeline(df, 
                              text_column='cleaned_text',
                              method='tfidf',
                              max_features=5000,
                              save_vectorizer=True):
    """
    Pipeline complet d'extraction de features
    
    Args:
        df: DataFrame contenant les textes
        text_column: Nom de la colonne de texte
        method: 'tfidf' ou 'word2vec'
        max_features: Nombre de features (pour TF-IDF)
        save_vectorizer: Sauvegarder le vectorizer
    
    Returns:
        X (features), vectorizer
    """
    print("="*60)
    print(" EXTRACTION DE FEATURES")
    print("="*60)
    
    texts = df[text_column].values
    
    if method == 'tfidf':
        extractor = CVFeatureExtractor(
            method='tfidf',
            max_features=max_features,
            ngram_range=(1, 2)
        )
        X = extractor.fit_transform(texts)
        
        if save_vectorizer:
            extractor.save()
        
        return X, extractor
    
    elif method == 'word2vec':
        extractor = Word2VecExtractor(vector_size=100)
        extractor.train(texts, epochs=10)
        X = extractor.transform(texts)
        
        return X, extractor
    
    else:
        raise ValueError(f"Méthode '{method}' non reconnue")


# ============================================
# SCRIPT PRINCIPAL
# ============================================
if __name__ == "__main__":
    print("="*60)
    print(" TEST D'EXTRACTION DE FEATURES")
    print("="*60)
    
    # Textes d'exemple
    sample_texts = [
        "python developer machine learning data science",
        "java software engineer backend development",
        "ui ux designer figma sketch adobe creative",
        "data scientist python sql tableau analytics",
        "frontend developer react javascript typescript"
    ]
    
    print("\n Test avec textes d'exemple...")
    
    # Test TF-IDF
    print("\n" + "="*60)
    print(" TEST TF-IDF")
    print("="*60)
    
    extractor_tfidf = CVFeatureExtractor(method='tfidf', max_features=50)
    X_tfidf = extractor_tfidf.fit_transform(sample_texts)
    
    print(f"\n Top 10 features:")
    top_features = extractor_tfidf.get_top_features(n=10)
    for i, feature in enumerate(top_features, 1):
        print(f"  {i:2d}. {feature}")
    
    # Test sur dataset réel (si disponible)
    print("\n" + "="*60)
    print(" TEST SUR DATASET RÉEL")
    print("="*60)
    
    try:
        df = pd.read_csv('data/processed/resume_cleaned.csv')
        print(f" Dataset chargé: {len(df)} CV")
        
        # Extraire les features
        X, extractor = extract_features_pipeline(
            df,
            text_column='cleaned_text',
            method='tfidf',
            max_features=5000
        )
        
        print(f"\n Features extraites:")
        print(f"   - Shape: {X.shape}")
        print(f"   - Type: {type(X)}")
        
        # Sauvegarder les features
        print(f"\n Sauvegarde des features...")
        from scipy.sparse import save_npz
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        save_npz('data/processed/X_features.npz', X)
        print(" Features sauvegardées dans data/processed/X_features.npz")
        
    except FileNotFoundError:
        print("\n Dataset nettoyé non trouvé")
        print(" Exécutez d'abord text_cleaner.py pour nettoyer les données")