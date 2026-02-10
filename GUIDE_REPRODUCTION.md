# Guide de Reproduction - CV Classifier

Ce guide explique comment reproduire la solution complète du classificateur de CV, de l'installation jusqu'à l'interface web.

---

## Table des Matières

1. [Structure du Projet](#1-structure-du-projet)
2. [Classification des Fichiers](#2-classification-des-fichiers)
3. [Prérequis](#3-prérequis)
4. [Installation](#4-installation)
5. [Ordre d'Exécution](#5-ordre-dexécution)
6. [Description des Fichiers Essentiels](#6-description-des-fichiers-essentiels)
7. [Fichiers d'Exploration (Notebooks)](#7-fichiers-dexploration-notebooks)
8. [Fichiers Legacy (Non Utilisés)](#8-fichiers-legacy-non-utilisés)
9. [Lancement de l'API](#9-lancement-de-lapi)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Structure du Projet

```
Projet_NLPfinal/
│
├── data/
│   ├── raw/
│   │   └── resume_dataset.csv       # Dataset brut (962 CVs)
│   ├── processed/                    # Données nettoyées (généré)
│   └── splits/                       # Indices train/test (généré)
│       ├── train_indices.json
│       ├── test_indices.json
│       └── split_metadata.json
│
├── src/
│   ├── preprocessing/
│   │   └── text_cleaner.py          # Nettoyage de texte
│   ├── training/
│   │   ├── __init__.py              # Module exports
│   │   ├── data_splitter.py         # Split des données brutes
│   │   ├── transformers.py          # Transformers sklearn
│   │   ├── pipeline_builder.py      # Construction du pipeline
│   │   ├── trainer.py               # Entraînement + CV
│   │   └── evaluator.py             # Évaluation finale
│   ├── database/
│   │   └── db_manager.py            # Gestion historique
│   ├── pdf_processing/
│   │   └── pdf_extractor.py         # Extraction PDF
│   └── skills_extraction/
│       └── skills_detector.py       # Détection compétences
│
├── scripts/
│   └── train_pipeline.py            # Script principal d'entraînement
│
├── models/                           # Modèles entraînés (généré)
│   ├── cv_classifier_pipeline.pkl   # Pipeline complet
│   ├── label_encoder.pkl            # Encodeur de labels
│   ├── cv_results.json              # Métriques CV
│   ├── test_evaluation.json         # Métriques test
│   └── training_metadata.json       # Métadonnées
│
├── api/
│   ├── main.py                      # API FastAPI
│   ├── config.py                    # Configuration
│   └── frontend/
│       └── index.html               # Interface web
│
├── tests/
│   ├── integration/
│   │   └── test_api.py              # Tests API
│   └── unit/
│       └── test_text_cleaner.py     # Tests unitaires
│
├── requirements.txt                  # Dépendances Python
└── GUIDE_REPRODUCTION.md            # Ce fichier
```

---

## 2. Classification des Fichiers

### Légende

| Catégorie | Symbole | Description |
|-----------|---------|-------------|
| **ESSENTIEL** | ✅ | Nécessaire pour la production |
| **EXPLORATION** | 🔬 | Notebooks pour exploration/développement |
| **LEGACY** | ⚠️ | Ancien code, non utilisé |
| **GÉNÉRÉ** | 📦 | Généré automatiquement |

---

### Vue Complète des Fichiers

```
Projet_NLPfinal/
│
├── 📁 data/
│   ├── 📁 raw/
│   │   └── ✅ resume_dataset.csv          # Dataset source (REQUIS)
│   ├── 📁 processed/
│   │   ├── 📦 resume_cleaned.csv          # Généré par notebooks (legacy)
│   │   ├── 📦 resume_cleaned_compact.csv  # Généré (legacy)
│   │   └── 📦 resume_with_stats.csv       # Généré (legacy)
│   └── 📁 splits/
│       ├── 📦 train_indices.json          # Généré par train_pipeline.py
│       ├── 📦 test_indices.json           # Généré par train_pipeline.py
│       └── 📦 split_metadata.json         # Généré par train_pipeline.py
│
├── 📁 src/
│   ├── 📁 preprocessing/
│   │   └── ✅ text_cleaner.py             # Nettoyage texte (UTILISÉ)
│   ├── 📁 training/
│   │   ├── ✅ __init__.py                 # Module exports
│   │   ├── ✅ data_splitter.py            # Split anti-leakage
│   │   ├── ✅ transformers.py             # Wrapper sklearn
│   │   ├── ✅ pipeline_builder.py         # Construction pipeline
│   │   ├── ✅ trainer.py                  # Entraînement + CV
│   │   └── ✅ evaluator.py                # Évaluation finale
│   ├── 📁 database/
│   │   └── ✅ db_manager.py               # Historique (optionnel)
│   ├── 📁 pdf_processing/
│   │   └── ✅ pdf_extractor.py            # Extraction PDF (optionnel)
│   └── 📁 skills_extraction/
│       └── ✅ skills_detector.py          # Détection skills (optionnel)
│
├── 📁 scripts/
│   ├── ✅ train_pipeline.py               # ⭐ SCRIPT PRINCIPAL
│   └── 📁 utils/
│       ├── ⚠️ check_models.py             # Legacy
│       ├── ⚠️ reprocess_full_dataset.py   # Legacy
│       └── ⚠️ train_optimized.py          # Legacy
│
├── 📁 notebooks/                          # 🔬 EXPLORATION UNIQUEMENT
│   ├── 🔬 01_EDA.ipynb                    # Analyse exploratoire
│   ├── 🔬 02_preprocessing.ipynb          # Tests preprocessing
│   ├── 🔬 03_feature_extraction.ipynb     # Tests TF-IDF
│   ├── 🔬 04_modeling.ipynb               # Tests modèles
│   ├── 🔬 05_evaluation.ipynb             # Tests évaluation
│   └── 🔬 06_API_testing.ipynb            # Tests API
│
├── 📁 models/                             # 📦 GÉNÉRÉ
│   ├── 📦 cv_classifier_pipeline.pkl      # Pipeline complet
│   ├── 📦 label_encoder.pkl               # Encodeur labels
│   ├── 📦 cv_results.json                 # Métriques CV
│   ├── 📦 test_evaluation.json            # Métriques test
│   ├── 📦 training_metadata.json          # Métadonnées
│   ├── ⚠️ best_model.pkl                  # Legacy (ancien modèle)
│   ├── ⚠️ tfidf_vectorizer.pkl            # Legacy (ancien vectorizer)
│   ├── ⚠️ Random_Forest_model.pkl         # Legacy
│   └── ⚠️ Gradient_Boosting_model.pkl     # Legacy
│
├── 📁 api/
│   ├── ✅ main.py                         # ⭐ API PRINCIPALE
│   ├── ⚠️ config.py                       # Legacy (non utilisé)
│   ├── ⚠️ diagnostic_api.py               # Legacy
│   ├── ⚠️ enhanced_endpoints.py           # Legacy
│   ├── ⚠️ models.py                       # Legacy
│   ├── ⚠️ predict_service.py              # Legacy
│   ├── ⚠️ test_api.py                     # Legacy (remplacé par tests/)
│   └── 📁 frontend/
│       ├── ✅ index.html                  # Interface principale
│       └── ⚠️ cv_classifier_final.html    # Legacy
│
├── 📁 tests/
│   ├── 📁 integration/
│   │   └── ✅ test_api.py                 # Tests API
│   └── 📁 unit/
│       └── ✅ test_text_cleaner.py        # Tests unitaires
│
├── 📁 docs/
│   ├── 🔬 Architecture_Projet.html        # Documentation
│   └── 🔬 mlflow_guide_complete.html      # Guide MLflow
│
├── ✅ requirements.txt                    # Dépendances
├── ✅ GUIDE_REPRODUCTION.md               # Ce fichier
├── ✅ README.md                           # Présentation projet
└── ✅ pytest.ini                          # Config tests
```

---

### Résumé par Catégorie

| Catégorie | Nombre | Action |
|-----------|--------|--------|
| ✅ Essentiel | 18 fichiers | Garder |
| 🔬 Exploration | 8 fichiers | Garder pour référence |
| ⚠️ Legacy | 15+ fichiers | Peuvent être supprimés |
| 📦 Généré | 10+ fichiers | Régénérés automatiquement |

---

## 3. Prérequis

### Logiciels requis

| Logiciel | Version minimale | Vérification |
|----------|------------------|--------------|
| Python | 3.10+ | `python --version` |
| pip | 21.0+ | `pip --version` |
| Git | 2.0+ | `git --version` |

### Dataset

Le fichier `data/raw/resume_dataset.csv` doit contenir:
- Colonne `Resume`: Texte brut du CV
- Colonne `Category`: Catégorie professionnelle (25 classes)

---

## 4. Installation

### Étape 1: Cloner le projet

```bash
git clone https://github.com/Tchapda3002/nlp-project.git
cd Projet_NLPfinal
```

### Étape 2: Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### Étape 3: Installer les dépendances

```bash
pip install -r requirements.txt
```

**Dépendances principales:**
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
nltk>=3.8.0
fastapi>=0.100.0
uvicorn>=0.23.0
joblib>=1.3.0
pdfplumber>=0.10.0      # Optionnel: extraction PDF
pytesseract>=0.3.10     # Optionnel: OCR
```

### Étape 4: Télécharger les ressources NLTK

```bash
python -c "
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
"
```

---

## 5. Ordre d'Exécution

### Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: DONNÉES                                               │
│  data/raw/resume_dataset.csv (manuel)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: ENTRAÎNEMENT                                          │
│  python scripts/train_pipeline.py                               │
│                                                                 │
│  Exécute automatiquement:                                       │
│  1. src/training/data_splitter.py    → Split train/test        │
│  2. src/training/transformers.py     → Nettoyage texte         │
│  3. src/training/pipeline_builder.py → Construction pipeline   │
│  4. src/training/trainer.py          → Cross-validation        │
│  5. src/training/evaluator.py        → Évaluation finale       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: API                                                   │
│  python -m uvicorn api.main:app --reload                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: INTERFACE                                             │
│  http://localhost:8000/app                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

### Commandes à exécuter dans l'ordre

```bash
# 1. S'assurer que le dataset est présent
ls data/raw/resume_dataset.csv

# 2. Entraîner le modèle (génère tout automatiquement)
python scripts/train_pipeline.py

# 3. Vérifier que les modèles sont générés
ls models/

# 4. Lancer les tests (optionnel mais recommandé)
python -m pytest tests/ -v

# 5. Démarrer l'API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 6. Accéder à l'interface
# Ouvrir: http://localhost:8000/app
```

---

## 6. Description des Fichiers Essentiels

### Phase 1: Données

| Fichier | Rôle | Quand l'utiliser |
|---------|------|------------------|
| `data/raw/resume_dataset.csv` | Dataset source avec 962 CVs | Doit exister avant l'entraînement |

---

### Phase 2: Entraînement

#### `scripts/train_pipeline.py` ⭐ POINT D'ENTRÉE PRINCIPAL

```bash
python scripts/train_pipeline.py [OPTIONS]
```

**Options:**
| Option | Défaut | Description |
|--------|--------|-------------|
| `--classifier` | random_forest | Algorithme (random_forest, gradient_boosting, logistic_regression, naive_bayes, svm) |
| `--test-size` | 0.2 | Proportion du test set |
| `--n-folds` | 5 | Nombre de folds pour la CV |
| `--skip-cv` | False | Ignorer la cross-validation |
| `--force-new-split` | False | Forcer un nouveau split |

**Ce qu'il fait:**
1. Charge `data/raw/resume_dataset.csv`
2. Split 80/20 sur données BRUTES (anti data-leakage)
3. Sauvegarde les indices dans `data/splits/`
4. Exécute une cross-validation 5-fold
5. Entraîne le modèle final
6. Évalue sur le test set
7. Sauvegarde tout dans `models/`

---

#### `src/training/data_splitter.py`

**Rôle:** Séparer les données AVANT tout preprocessing

**Pourquoi c'est important:**
- Évite la fuite de données (data leakage)
- Le test set ne doit JAMAIS influencer l'entraînement
- Sauvegarde les indices pour reproductibilité

**Classe principale:**
```python
class DataSplitter:
    def split_and_save(df, target_column, output_dir)
    def load_split(df, split_dir)
    def split_exists(split_dir)
```

---

#### `src/training/transformers.py`

**Rôle:** Wrapper sklearn autour de TextCleaner

**Pourquoi c'est important:**
- Permet d'intégrer le nettoyage dans un Pipeline sklearn
- Le nettoyage est appliqué APRÈS le split
- Stateless: fit() ne fait rien, transform() nettoie

**Classe principale:**
```python
class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    def fit(X, y=None)      # Ne fait rien (stateless)
    def transform(X)         # Nettoie le texte
```

---

#### `src/training/pipeline_builder.py`

**Rôle:** Construire le pipeline sklearn complet

**Structure du pipeline:**
```
TextCleanerTransformer → TfidfVectorizer → Classifier
```

**Pourquoi c'est important:**
- Encapsule TOUTES les transformations
- Garantit que TF-IDF est fit UNIQUEMENT sur train
- Facilite la prédiction (une seule ligne)

**Classe principale:**
```python
class CVClassifierPipelineBuilder:
    def build()              # Retourne un Pipeline sklearn
    def get_param_grid()     # Pour GridSearchCV
```

---

#### `src/training/trainer.py`

**Rôle:** Orchestrer l'entraînement et la cross-validation

**Pourquoi c'est important:**
- Cross-validation sur train UNIQUEMENT
- Mesure la vraie performance de généralisation
- Sauvegarde les métriques pour traçabilité

**Classe principale:**
```python
class CVClassifierTrainer:
    def cross_validate(X_train, y_train)  # CV 5-fold
    def train(X_train, y_train)           # Entraînement final
    def save(output_dir)                  # Sauvegarde
```

---

#### `src/training/evaluator.py`

**Rôle:** Évaluation finale sur le test set

**Pourquoi c'est important:**
- Appelé UNE SEULE FOIS à la fin
- Donne la vraie performance sur données jamais vues
- Compare avec les résultats de CV

**Classe principale:**
```python
class PipelineEvaluator:
    def evaluate(X_test, y_test)       # Métriques complètes
    def compare_with_cv(cv_results)    # Détection overfitting
    def save_report(output_dir)        # Rapport JSON + TXT
```

---

#### `src/preprocessing/text_cleaner.py`

**Rôle:** Nettoyage du texte des CVs

**Transformations:**
1. Mise en minuscules
2. Suppression URLs, emails, téléphones
3. Suppression ponctuation
4. Tokenisation
5. Suppression stopwords
6. Lemmatisation

**Classe principale:**
```python
class TextCleaner:
    def clean_text(text)           # Nettoie un texte
    def clean_dataframe(df, col)   # Nettoie une colonne
```

---

### Phase 3: API

#### `api/main.py` ⭐ API PRINCIPALE

**Rôle:** Exposer le modèle via REST API

**Endpoints principaux:**

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Informations API |
| `/health` | GET | État de santé |
| `/predict` | POST | Classifier un CV |
| `/model-info` | GET | Métriques du modèle |
| `/categories` | GET | Liste des catégories |
| `/upload-cv` | POST | Upload PDF + classification |
| `/analyze-skills` | POST | Extraction compétences |
| `/app` | GET | Interface web |

**Chargement automatique:**
```python
# Charge dans cet ordre:
1. cv_classifier_pipeline.pkl   # Pipeline complet
2. label_encoder.pkl            # Décodage labels
3. cv_results.json              # Métriques CV
4. test_evaluation.json         # Métriques test
5. training_metadata.json       # Métadonnées
```

---

#### `api/frontend/index.html`

**Rôle:** Interface web interactive

**Fonctionnalités:**
- Upload de CV (texte ou PDF)
- Options d'analyse (skills, expérience, recommandations)
- Affichage des probabilités top 5
- Historique des classifications

---

### Phase 4: Tests

#### `tests/integration/test_api.py`

**Rôle:** Tester les endpoints API

```bash
python -m pytest tests/integration/ -v
```

#### `tests/unit/test_text_cleaner.py`

**Rôle:** Tester le nettoyage de texte

```bash
python -m pytest tests/unit/ -v
```

---

## 7. Fichiers d'Exploration (Notebooks)

Les notebooks sont utilisés pour l'**exploration et le développement**, mais ne sont **PAS nécessaires** pour la production.

### Quand utiliser les notebooks ?

| Notebook | Utilité | Quand l'exécuter |
|----------|---------|------------------|
| `01_EDA.ipynb` | Analyse exploratoire des données | Pour comprendre le dataset |
| `02_preprocessing.ipynb` | Tester le nettoyage de texte | Pour ajuster les paramètres de nettoyage |
| `03_feature_extraction.ipynb` | Tester TF-IDF | Pour optimiser les hyperparamètres |
| `04_modeling.ipynb` | Comparer différents modèles | Pour choisir le meilleur algorithme |
| `05_evaluation.ipynb` | Analyser les erreurs | Pour comprendre les faiblesses du modèle |
| `06_API_testing.ipynb` | Tester l'API manuellement | Pour debug |

### Ordre d'exécution des notebooks (si nécessaire)

```
01_EDA.ipynb
     │
     ▼
02_preprocessing.ipynb
     │
     ▼
03_feature_extraction.ipynb
     │
     ▼
04_modeling.ipynb
     │
     ▼
05_evaluation.ipynb
     │
     ▼
06_API_testing.ipynb (après avoir lancé l'API)
```

### Important

> ⚠️ Les notebooks peuvent contenir du code **avec fuite de données** (data leakage) car ils ont été créés pendant la phase d'exploration.
>
> Pour l'entraînement final, utilisez **TOUJOURS** `scripts/train_pipeline.py` qui implémente les bonnes pratiques anti-leakage.

---

## 8. Fichiers Legacy (Non Utilisés)

Ces fichiers ont été créés pendant le développement mais ne sont **plus utilisés** dans le workflow actuel.

### Fichiers à supprimer (optionnel)

```bash
# Scripts legacy
rm scripts/utils/check_models.py
rm scripts/utils/reprocess_full_dataset.py
rm scripts/utils/train_optimized.py

# API legacy
rm api/config.py
rm api/diagnostic_api.py
rm api/enhanced_endpoints.py
rm api/models.py
rm api/predict_service.py
rm api/test_api.py

# Frontend legacy
rm api/frontend/cv_classifier_final.html

# Anciens modèles (remplacés par le pipeline)
rm models/best_model.pkl
rm models/tfidf_vectorizer.pkl
rm models/Random_Forest_model.pkl
rm models/Gradient_Boosting_model.pkl

# Données processées (le pipeline les régénère)
rm data/processed/resume_cleaned.csv
rm data/processed/resume_cleaned_compact.csv
rm data/processed/resume_with_stats.csv
```

### Pourquoi ces fichiers existent ?

| Fichier | Historique |
|---------|------------|
| `best_model.pkl` | Ancien modèle entraîné SANS split correct |
| `tfidf_vectorizer.pkl` | Vectorizer fit sur TOUTES les données (leakage) |
| `api/config.py` | Configuration non utilisée |
| `scripts/utils/*.py` | Anciens scripts remplacés par `train_pipeline.py` |

---

## 9. Lancement de l'API

### Développement

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Accès

| URL | Description |
|-----|-------------|
| http://localhost:8000 | Racine API |
| http://localhost:8000/app | Interface web |
| http://localhost:8000/docs | Documentation Swagger |
| http://localhost:8000/redoc | Documentation ReDoc |

---

## 10. Troubleshooting

### Erreur: "Module not found"

```bash
# Vérifier que vous êtes dans le bon dossier
pwd  # Doit afficher .../Projet_NLPfinal

# Réinstaller les dépendances
pip install -r requirements.txt
```

### Erreur: "Dataset not found"

```bash
# Vérifier que le dataset existe
ls -la data/raw/resume_dataset.csv

# Si manquant, placer le fichier CSV dans data/raw/
```

### Erreur: "Pipeline not found"

```bash
# Réentraîner le modèle
python scripts/train_pipeline.py
```

### Erreur: "Port already in use"

```bash
# Tuer le processus existant
pkill -f "uvicorn.*main:app"

# Ou utiliser un autre port
python -m uvicorn api.main:app --port 8001
```

### Erreur NLTK

```bash
python -c "
import nltk
nltk.download('all')
"
```

---

## Résumé: Commandes Essentielles

```bash
# Installation complète
pip install -r requirements.txt

# Entraînement
python scripts/train_pipeline.py

# Tests
python -m pytest tests/ -v

# Lancement API
python -m uvicorn api.main:app --reload

# Accès interface
open http://localhost:8000/app
```

---

## Métriques Attendues

Après entraînement, vous devriez obtenir:

| Métrique | Cross-Validation | Test Set |
|----------|------------------|----------|
| Accuracy | ~99.2% | ~100% |
| F1 Macro | ~99.0% | ~100% |
| Precision | ~99.3% | ~100% |
| Recall | ~98.9% | ~100% |

Ces métriques sont sauvegardées dans `models/` et chargées dynamiquement par l'API.
