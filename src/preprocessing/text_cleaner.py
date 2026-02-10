"""
src/preprocessing/text_cleaner.py
Module pour le nettoyage et prétraitement du texte des CV
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
from tqdm import tqdm

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class TextCleaner:
    """Classe pour nettoyer et prétraiter le texte des CV"""
    
    def __init__(self, 
                 lowercase=True,
                 remove_urls=True,
                 remove_emails=True,
                 remove_phone_numbers=True,
                 remove_numbers=True,
                 remove_punctuation=True,
                 remove_stopwords=True,
                 lemmatize=True,
                 stem=False):
        """
        Initialiser le nettoyeur de texte
        
        Args:
            lowercase: Convertir en minuscules
            remove_urls: Supprimer les URLs
            remove_emails: Supprimer les emails
            remove_phone_numbers: Supprimer les numéros de téléphone
            remove_numbers: Supprimer les nombres
            remove_punctuation: Supprimer la ponctuation
            remove_stopwords: Supprimer les mots vides (stopwords)
            lemmatize: Appliquer la lemmatisation
            stem: Appliquer le stemming (si False et lemmatize=True, on lemmatise)
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_phone_numbers = remove_phone_numbers
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        
        # Initialiser les outils NLTK
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
    def clean_text(self, text):
        """
        Nettoyer un texte unique
        
        Args:
            text: Texte à nettoyer (string)
        
        Returns:
            Texte nettoyé (string)
        """
        if not isinstance(text, str):
            text = str(text)
        
        # 1. Convertir en minuscules
        if self.lowercase:
            text = text.lower()
        
        # 2. Supprimer les URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 3. Supprimer les emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # 4. Supprimer les numéros de téléphone
        if self.remove_phone_numbers:
            # Pattern pour téléphones US, UK, etc.
            text = re.sub(r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '', text)
        
        # 5. Supprimer les nombres
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # 6. Supprimer la ponctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 7. Supprimer les espaces multiples et nettoyer
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 8. Tokenisation
        tokens = word_tokenize(text)
        
        # 9. Supprimer les stopwords
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        # 10. Supprimer les tokens trop courts (< 2 caractères)
        tokens = [word for word in tokens if len(word) > 2]
        
        # 11. Lemmatisation ou Stemming
        if self.lemmatize and not self.stem:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        elif self.stem:
            tokens = [self.stemmer.stem(word) for word in tokens]
        
        # 12. Rejoindre les tokens
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def clean_dataframe(self, df, text_column='Resume', output_column='cleaned_text'):
        """
        Nettoyer une colonne de texte dans un DataFrame
        
        Args:
            df: DataFrame pandas
            text_column: Nom de la colonne contenant le texte
            output_column: Nom de la colonne pour le texte nettoyé
        
        Returns:
            DataFrame avec colonne de texte nettoyé
        """
        print(f" Nettoyage de {len(df)} textes...")
        
        # Appliquer le nettoyage avec barre de progression
        tqdm.pandas(desc="Nettoyage en cours")
        df[output_column] = df[text_column].progress_apply(self.clean_text)
        
        # Statistiques
        avg_length_before = df[text_column].astype(str).apply(len).mean()
        avg_length_after = df[output_column].apply(len).mean()
        
        print(f"\n Nettoyage terminé!")
        print(f" Longueur moyenne avant: {avg_length_before:.0f} caractères")
        print(f" Longueur moyenne après: {avg_length_after:.0f} caractères")
        print(f" Réduction: {((avg_length_before - avg_length_after) / avg_length_before * 100):.1f}%")
        
        return df
    
    def get_word_frequency(self, texts, top_n=20):
        """
        Obtenir les mots les plus fréquents
        
        Args:
            texts: Liste de textes ou Series pandas
            top_n: Nombre de mots les plus fréquents à retourner
        
        Returns:
            Dictionary avec les mots et leurs fréquences
        """
        from collections import Counter
        
        # Combiner tous les textes
        all_words = []
        for text in texts:
            all_words.extend(text.split())
        
        # Compter les fréquences
        word_freq = Counter(all_words)
        
        return dict(word_freq.most_common(top_n))
    
    def visualize_cleaning_effect(self, original_text, cleaned_text):
        """
        Afficher l'effet du nettoyage sur un texte
        
        Args:
            original_text: Texte original
            cleaned_text: Texte nettoyé
        """
        print("\n" + "="*60)
        print(" COMPARAISON AVANT/APRÈS NETTOYAGE")
        print("="*60)
        
        print(f"\n TEXTE ORIGINAL ({len(original_text)} caractères):")
        print("-" * 60)
        print(original_text[:500] + "..." if len(original_text) > 500 else original_text)
        
        print(f"\n TEXTE NETTOYÉ ({len(cleaned_text)} caractères):")
        print("-" * 60)
        print(cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text)
        
        print(f"\n STATISTIQUES:")
        print(f"  - Caractères originaux: {len(original_text)}")
        print(f"  - Caractères nettoyés: {len(cleaned_text)}")
        print(f"  - Réduction: {len(original_text) - len(cleaned_text)} caractères ({(1 - len(cleaned_text)/len(original_text))*100:.1f}%)")
        print(f"  - Mots originaux: {len(original_text.split())}")
        print(f"  - Mots nettoyés: {len(cleaned_text.split())}")


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def clean_cv_dataset(input_path='data/raw/resume_dataset.csv',
                     output_path='data/processed/resume_cleaned.csv',
                     text_column='Resume'):
    """
    Nettoyer un dataset complet de CV et sauvegarder
    
    Args:
        input_path: Chemin du fichier CSV d'entrée
        output_path: Chemin du fichier CSV de sortie
        text_column: Nom de la colonne contenant le texte
    
    Returns:
        DataFrame nettoyé
    """
    # Charger les données
    print(f" Chargement de {input_path}...")
    df = pd.read_csv(input_path)
    print(f" {len(df)} CV chargés")
    
    # Créer le nettoyeur
    cleaner = TextCleaner(
        lowercase=True,
        remove_urls=True,
        remove_emails=True,
        remove_phone_numbers=True,
        remove_numbers=False,  # Garder les nombres (années d'expérience, etc.)
        remove_punctuation=True,
        remove_stopwords=True,
        lemmatize=True,
        stem=False
    )
    
    # Nettoyer
    df = cleaner.clean_dataframe(df, text_column=text_column)
    
    # Sauvegarder
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n Dataset nettoyé sauvegardé dans {output_path}")
    
    return df


# ============================================
# SCRIPT PRINCIPAL
# ============================================
if __name__ == "__main__":
    print("="*60)
    print(" NETTOYAGE DU TEXTE DES CV")
    print("="*60)
    
    # Exemple avec un texte unique
    sample_text = """
    John Doe
    Email: john.doe@example.com | Phone: +1-555-123-4567
    Website: https://johndoe.com
    
    PROFESSIONAL SUMMARY
    Experienced Software Developer with 5+ years in Python, JavaScript, and cloud technologies.
    Strong background in machine learning and data analysis. Looking for new opportunities!!!
    
    EXPERIENCE
    - Senior Developer at TechCorp (2020-2023)
    - Junior Developer at StartupXYZ (2018-2020)
    
    SKILLS: Python, JavaScript, React, Docker, AWS, SQL, MongoDB
    """
    
    print("\n Test de nettoyage sur un texte exemple...")
    
    # Créer le nettoyeur
    cleaner = TextCleaner()
    
    # Nettoyer le texte
    cleaned = cleaner.clean_text(sample_text)
    
    # Visualiser l'effet
    cleaner.visualize_cleaning_effect(sample_text, cleaned)
    
    # Test sur dataset complet (si disponible)
    print("\n" + "="*60)
    print(" NETTOYAGE DU DATASET COMPLET")
    print("="*60)
    
    try:
        df_cleaned = clean_cv_dataset(
            input_path='data/raw/resume_dataset.csv',
            output_path='data/processed/resume_cleaned.csv'
        )
        
        # Afficher les mots les plus fréquents
        print("\n Top 20 mots les plus fréquents après nettoyage:")
        word_freq = cleaner.get_word_frequency(df_cleaned['cleaned_text'], top_n=20)
        for i, (word, freq) in enumerate(word_freq.items(), 1):
            print(f"  {i:2d}. {word:20s} : {freq:5d}")
            
    except FileNotFoundError:
        print("\n Fichier data/raw/resume_dataset.csv non trouvé")
        print(" Placez votre dataset dans ce chemin pour continuer")