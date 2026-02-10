#  Guide de Démarrage Rapide - CV Classifier Pro

## Installation en 5 Minutes

### 1. Cloner le Projet

```bash
git clone https://github.com/votre-username/cv-classifier-pro.git
cd cv-classifier-pro
```

### 2. Créer l'Environnement Virtuel

```bash
# Créer
python -m venv venv

# Activer (Windows)
venv\Scripts\activate

# Activer (Linux/Mac)
source venv/bin/activate
```

### 3. Installer les Dépendances

```bash
pip install -r requirements.txt
```

### 4. Installer Tesseract (pour OCR)

**Windows:**
- Télécharger: https://github.com/UB-Mannheim/tesseract/wiki
- Installer et ajouter au PATH

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

### 5. Lancer l'API

```bash
cd api
python main.py
```

L'API démarre sur: http://localhost:8000

### 6. Ouvrir l'Interface

Ouvrir dans un navigateur:
```
frontend/cv_classifier_batch.html
```

##  Vérification

### Tester l'API

```bash
# Health check
curl http://localhost:8000/health

# Test de prédiction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Python developer with 5 years experience"}'
```

### Documentation Interactive

Accéder à: http://localhost:8000/docs

##  MLflow (Optionnel)

```bash
mlflow ui --port 5000
```

Accéder à: http://localhost:5000

##  Premier Projet

### Structure Minimale Requise

```
cv-classifier-pro/
├── api/
│   └── main.py                  Requis
├── src/
│   ├── database/
│   │   └── db_manager.py        Requis
│   ├── pdf_processing/
│   │   └── pdf_extractor.py     Requis
│   ├── skills_extraction/
│   │   └── skills_detector.py   Requis
│   └── preprocessing/
│       └── text_cleaner.py      Requis
├── models_saved/
│   ├── best_model.pkl           Requis
│   ├── tfidf_vectorizer.pkl     Requis
│   └── label_encoder.pkl        Requis
├── data/                        Se crée auto
├── frontend/
│   └── cv_classifier_batch.html  Requis
└── requirements.txt             Requis
```

##  Problèmes Courants

### Erreur: Module not found

```bash
pip install -r requirements.txt --force-reinstall
```

### Erreur: Tesseract not found

Ajouter au PATH (Windows):
```
C:\Program Files\Tesseract-OCR
```

### Erreur: Port 8000 already in use

```bash
# Changer le port dans api/main.py
uvicorn.run("main:app", port=8001)
```

### Erreur: scikit-learn version mismatch

```bash
pip install scikit-learn==1.3.2
```

##  Prochaines Étapes

1.  Installer et tester l'API
2.  Entraîner vos propres modèles (notebooks/)
3.  Personnaliser l'interface (frontend/)
4.  Configurer MLflow pour le tracking
5.  Déployer en production

##  Commandes Utiles

```bash
# Voir les packages installés
pip list

# Mettre à jour les dépendances
pip install --upgrade -r requirements.txt

# Créer un export des packages
pip freeze > requirements-freeze.txt

# Lancer les tests
pytest tests/

# Formatter le code
black src/ api/

# Linter
pylint src/ api/
```

##  Support

- Documentation: [README.md](README.md)
- Issues: https://github.com/votre-username/cv-classifier-pro/issues
- Email: votre.email@example.com

---

**Bon développement ! **
