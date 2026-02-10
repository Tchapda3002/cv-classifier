# CV Classifier Professional

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Application professionnelle de classification automatique de CV utilisant l'intelligence artificielle et le Machine Learning.

##  Table des Matières

- [Fonctionnalités](#-fonctionnalités)
- [Technologies](#-technologies)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Architecture](#-architecture)
- [API Documentation](#-api-documentation)
- [MLflow Integration](#-mlflow-integration)
- [Contribuer](#-contribuer)
- [License](#-license)

##  Fonctionnalités

### Classification ML
- Classification automatique de CV en 25+ catégories professionnelles
- Modèles ML : Random Forest, SVM, Logistic Regression, Naive Bayes
- Confiance de prédiction et probabilités pour toutes les catégories
- Support multi-formats : PDF, DOCX, TXT, LaTeX (.tex)

### Extraction Avancée
- **Extraction PDF** : Support PDFs natifs et scannés (OCR)
- **Extraction Word** : Documents .docx
- **Extraction LaTeX** : Fichiers .tex
- **Détection automatique** du type de document

### Analyse de Compétences
- Détection automatique de 1000+ compétences techniques
- Extraction de soft skills, frameworks, outils, langages
- Analyse de l'expérience (années, niveau)
- Identification des certifications

### Recommandations
- Recommandations de postes basées sur les compétences
- Score de correspondance pour chaque poste
- Identification des lacunes de compétences
- Niveau d'expérience requis (Junior/Mid/Senior)

### Base de Données
- Historique complet des classifications
- Statistiques en temps réel
- Feedback utilisateur
- Export CSV/Excel

### Analyse Batch
- Upload et analyse de plusieurs CVs simultanément
- Traitement séquentiel avec progression en temps réel
- Résumé global (Total/Réussi/Échoué)
- Résultats individuels par CV

### Interface Web
- Interface moderne et responsive
- Upload drag & drop multi-fichiers
- Paste de texte direct
- Visualisations interactives
- Design professionnel sans émojis

### MLflow Integration
- Tracking de tous les entraînements
- Comparaison des modèles
- Versioning automatique
- Métriques et artifacts

##  Technologies

### Backend
- **Python 3.8+**
- **FastAPI** - API REST moderne
- **scikit-learn** - Machine Learning
- **MLflow** - ML Lifecycle Management
- **SQLite** - Base de données

### Extraction & NLP
- **pdfplumber** - Extraction PDF
- **pytesseract** - OCR pour PDFs scannés
- **python-docx** - Extraction Word
- **NLTK** - Traitement du langage naturel

### Frontend
- **HTML5/CSS3/JavaScript**
- **Font: Inter** - Typographie moderne
- **Responsive Design** - Mobile-friendly

##  Installation

### Prérequis

```bash
# Python 3.8 ou supérieur
python --version

# Git
git --version
```

### Installation Tesseract (pour OCR)

**Windows:**
```bash
# Télécharger depuis: https://github.com/UB-Mannheim/tesseract/wiki
# Ajouter au PATH: C:\Program Files\Tesseract-OCR
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### Cloner le Projet

```bash
git clone https://github.com/votre-username/cv-classifier-pro.git
cd cv-classifier-pro
```

### Créer un Environnement Virtuel

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Installer les Dépendances

```bash
pip install -r requirements.txt
```

### Structure du Projet

```
cv-classifier-pro/
├── api/
│   ├── main.py                    # API FastAPI
│   └── enhanced_endpoints.py      # Endpoints avancés
├── src/
│   ├── database/
│   │   └── db_manager.py          # Gestionnaire DB
│   ├── pdf_processing/
│   │   └── pdf_extractor.py       # Extracteur PDF
│   ├── skills_extraction/
│   │   └── skills_detector.py     # Détecteur compétences
│   └── preprocessing/
│       └── text_cleaner.py        # Nettoyage texte
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb          # Entraînement modèles
├── data/
│   ├── raw/                       # Données brutes
│   ├── processed/                 # Données traitées
│   └── cv_history.db              # Base de données
├── models_saved/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
├── outputs/
│   ├── plots/                     # Visualisations
│   ├── reports/                   # Rapports
│   └── exports/                   # Exports CSV
├── frontend/
│   └── cv_classifier_batch.html   # Interface web
├── .gitignore
├── requirements.txt
└── README.md
```

##  Utilisation

### 1. Lancer MLflow (optionnel)

```bash
mlflow ui --port 5000
# Accéder à: http://localhost:5000
```

### 2. Lancer l'API

```bash
cd api
python main.py
# API disponible sur: http://localhost:8000
```

### 3. Accéder à l'Interface

Ouvrir dans un navigateur:
```
frontend/cv_classifier_batch.html
```

Ou accéder à la documentation API:
```
http://localhost:8000/docs
```

### 4. Entraîner les Modèles

```bash
cd notebooks
jupyter notebook 04_modeling.ipynb
# Exécuter toutes les cellules
```

##  Architecture

### Pipeline ML

```
CV Input
   ↓
[Extraction] → PDF/DOCX/TXT/TEX → Texte brut
   ↓
[Nettoyage] → Suppression stop words, lemmatisation
   ↓
[Vectorisation] → TF-IDF (5000 features)
   ↓
[Classification] → Random Forest / SVM / Logistic Regression
   ↓
[Résultat] → Catégorie + Confiance
   ↓
[Analyse] → Compétences + Recommandations
   ↓
[Stockage] → Base de données SQLite
```

### API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Page d'accueil |
| `/health` | GET | État de l'API |
| `/predict` | POST | Classification texte |
| `/upload-cv` | POST | Upload PDF complet |
| `/analyze-skills` | POST | Analyse compétences |
| `/history` | GET | Historique |
| `/statistics` | GET | Statistiques |
| `/export` | POST | Export CSV |
| `/categories` | GET | Liste catégories |

##  API Documentation

### Classification Simple

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Experienced Python developer with 5 years in ML"
  }'
```

**Réponse:**
```json
{
  "category": "Data Scientist",
  "confidence": 0.95,
  "all_probabilities": {
    "Data Scientist": 0.95,
    "ML Engineer": 0.03,
    "Software Engineer": 0.02
  }
}
```

### Upload CV Complet

```bash
curl -X POST "http://localhost:8000/upload-cv?extract_skills=true&recommend_jobs=true" \
  -F "file=@cv.pdf"
```

**Réponse:**
```json
{
  "classification_id": 42,
  "filename": "cv.pdf",
  "predicted_category": "Data Scientist",
  "confidence": 0.95,
  "skills_summary": {
    "total_technical_skills": 18,
    "total_frameworks": 8,
    "total_tools": 12
  },
  "job_recommendations": [
    {
      "job_title": "Data Scientist",
      "match_score": 0.92,
      "experience_level_fit": "Good fit"
    }
  ]
}
```

### Statistiques

```bash
curl "http://localhost:8000/statistics"
```

**Réponse:**
```json
{
  "total_classifications": 150,
  "avg_confidence": 0.89,
  "category_distribution": [
    {"category": "Data Scientist", "count": 45},
    {"category": "Software Engineer", "count": 38}
  ],
  "top_skills": [
    {"skill": "Python", "count": 89},
    {"skill": "Machine Learning", "count": 67}
  ]
}
```

##  MLflow Integration

### Lancer MLflow

```bash
mlflow ui --port 5000
```

### Tracker un Entraînement

```python
import mlflow

mlflow.set_experiment("CV_Classification")

with mlflow.start_run(run_name="Random_Forest"):
    # Entraîner le modèle
    model.fit(X_train, y_train)
    
    # Logger les métriques
    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1_score": f1
    })
    
    # Logger le modèle
    mlflow.sklearn.log_model(model, "model")
```

### Comparer les Modèles

1. Accéder à http://localhost:5000
2. Sélectionner plusieurs runs
3. Cliquer sur "Compare"
4. Visualiser les métriques

##  Interface Web

### Fonctionnalités

- Upload drag & drop multi-fichiers
- Paste de texte direct
- Options d'analyse configurables
- Résultats en temps réel
- Statistiques live
- Design responsive

### Utilisation

1. **Upload Files**: Glissez-déposez un ou plusieurs CVs
2. **Configure**: Activez les options (skills, recommendations, history)
3. **Analyze**: Cliquez sur "Analyze X CVs"
4. **Results**: Visualisez les résultats pour chaque CV

##  Performance

### Métriques Modèle

| Modèle | Accuracy | F1-Score | Training Time |
|--------|----------|----------|---------------|
| Random Forest | 94.2% | 93.8% | 2.3s |
| SVM | 92.1% | 91.7% | 4.1s |
| Logistic Regression | 89.5% | 88.9% | 0.8s |

### Catégories Supportées (25)

- Data Scientist
- Software Engineer
- ML Engineer
- Web Developer
- DevOps Engineer
- Product Manager
- Business Analyst
- [... et 18 autres]

##  Contribuer

Les contributions sont les bienvenues !

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amazing-feature`)
3. Commit vos changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

### Guidelines

- Code Python conforme à PEP 8
- Tests unitaires pour nouvelles fonctionnalités
- Documentation claire
- Commits descriptifs

##  License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

##  Auteurs

- **Saer NDAO** - *Développement initial* - [YourGitHub](https://github.com/ndaosaer)

##  Remerciements

- scikit-learn pour les algorithmes ML
- FastAPI pour le framework web
- MLflow pour le tracking ML
- Tesseract pour l'OCR

##  Support

Pour toute question ou problème:
- Ouvrir une [issue](https://github.com/ndaosaer/Projet_NLPFinal/issues)
- Email: saerndao469@gmail.com

##  Roadmap

- [ ] Support de nouvelles langues (FR, ES, DE)
- [ ] Dashboard analytics avancé
- [ ] Export PDF des résultats
- [ ] API authentification
- [ ] Déploiement cloud (AWS/Azure)
- [ ] Application mobile

---

**Made with using Python and FastAPI**
"# Projet_NLPFinal" 
