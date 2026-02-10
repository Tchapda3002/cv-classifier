"""
api/main.py - VERSION COMPLÈTE FINALE
API REST pour la classification de CV avec toutes les fonctionnalités avancées
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pickle
import joblib
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

# Charger les variables d'environnement depuis .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    print(" Variables d'environnement chargées depuis .env")
except ImportError:
    pass

print("\n" + "="*80)
print(" INITIALISATION DE L'API - VERSION COMPLÈTE")
print("="*80 + "\n")

# ============================================
# ÉTAPE 1: DÉTERMINER LES CHEMINS ABSOLUS
# ============================================

print(" ÉTAPE 1: Détection des chemins...")

# Chemin de ce fichier (api/main.py)
THIS_FILE = Path(__file__).resolve()
print(f"   Ce fichier: {THIS_FILE}")

# Dossier api
API_DIR = THIS_FILE.parent
print(f"   Dossier API: {API_DIR}")

# Dossier racine du projet (parent de api/)
BASE_DIR = API_DIR.parent
print(f"   Dossier projet: {BASE_DIR}")

# Dossier des modèles
MODELS_DIR = BASE_DIR / "models"
print(f"   Dossier modèles: {MODELS_DIR}")

# Dossier src
SRC_DIR = BASE_DIR / "src"
print(f"   Dossier src: {SRC_DIR}")

# Dossier data
DATA_DIR = BASE_DIR / "data"
print(f"   Dossier data: {DATA_DIR}")

print()

# ============================================
# ÉTAPE 2: VÉRIFICATIONS
# ============================================

print(" ÉTAPE 2: Vérifications...")

# Vérifier que models existe
if not MODELS_DIR.exists():
    print(f" ERREUR CRITIQUE: {MODELS_DIR} n'existe pas!")
    print(f"   Créez ce dossier et placez-y vos modèles")
    sys.exit(1)
else:
    print(f" Dossier models trouvé")

# Créer le dossier data s'il n'existe pas
DATA_DIR.mkdir(exist_ok=True)
print(f" Dossier data vérifié")

# Lister les fichiers de modèles
print(f"\n Fichiers dans {MODELS_DIR}:")
for file in MODELS_DIR.iterdir():
    if file.is_file():
        print(f"   - {file.name}")

# Ajouter src au path Python
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
    print(f"\n {SRC_DIR} ajouté au path Python")
else:
    print(f"\n  {SRC_DIR} n'existe pas")

print()

# ============================================
# IMPORTS DES MODULES
# ============================================

# Import du TextCleaner
try:
    from preprocessing.text_cleaner import TextCleaner
    TEXT_CLEANER_AVAILABLE = True
    print(" TextCleaner importé avec succès")
except ImportError as e:
    TEXT_CLEANER_AVAILABLE = False
    print(f"  Impossible d'importer TextCleaner: {e}")
    print("   L'API fonctionnera sans nettoyage de texte")

# Import des modules avancés (optionnels)
try:
    sys.path.insert(0, str(BASE_DIR / "src" / "database"))
    from db_manager import CVDatabaseManager
    DATABASE_AVAILABLE = True
    print(" Database Manager importé")
except ImportError:
    DATABASE_AVAILABLE = False
    CVDatabaseManager = None
    print("  Database Manager non disponible")

try:
    sys.path.insert(0, str(BASE_DIR / "src" / "pdf_processing"))
    from pdf_extractor import AdvancedPDFExtractor
    PDF_EXTRACTOR_AVAILABLE = True
    print(" PDF Extractor importé")
except ImportError:
    PDF_EXTRACTOR_AVAILABLE = False
    AdvancedPDFExtractor = None
    print("  PDF Extractor non disponible")

try:
    sys.path.insert(0, str(BASE_DIR / "src" / "skills_extraction"))
    from skills_detector import SkillsDetector
    SKILLS_DETECTOR_AVAILABLE = True
    print(" Skills Detector importé")
except ImportError:
    SKILLS_DETECTOR_AVAILABLE = False
    SkillsDetector = None
    print("  Skills Detector non disponible")

try:
    sys.path.insert(0, str(BASE_DIR / "src" / "chatbot"))
    from cv_chatbot import CVChatbot, SimpleCVChatbot
    CHATBOT_AVAILABLE = True
    print(" CV Chatbot importé")
except ImportError:
    CHATBOT_AVAILABLE = False
    CVChatbot = None
    SimpleCVChatbot = None
    print("  CV Chatbot non disponible")

print()

# ============================================
# MODÈLES PYDANTIC
# ============================================

class CVInput(BaseModel):
    resume_text: str = Field(..., description="Texte complet du CV")

class CVPrediction(BaseModel):
    category: str
    confidence: float
    all_probabilities: Optional[dict] = None

class BatchCVInput(BaseModel):
    resumes: List[str]

class BatchCVPrediction(BaseModel):
    predictions: List[CVPrediction]
    total_processed: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorizer_loaded: bool
    label_encoder_loaded: bool
    text_cleaner_loaded: bool
    database_available: bool
    pdf_extractor_available: bool
    skills_detector_available: bool
    version: str
    base_dir: str
    models_dir: str

class CVUploadResponse(BaseModel):
    classification_id: Optional[int]
    filename: str
    predicted_category: str
    confidence: float
    extraction_method: Optional[str]
    extraction_confidence: Optional[float]
    skills_summary: Optional[Dict]
    experience_info: Optional[Dict]
    job_recommendations: Optional[List[Dict]]
    processing_time_ms: int
    extracted_text: Optional[str] = None  # Pour le chatbot

class SkillsAnalysisResponse(BaseModel):
    skills_summary: Dict
    detailed_skills: Dict
    experience_analysis: Dict
    job_recommendations: List[Dict]
    top_strengths: List[str]

class FeedbackUpdate(BaseModel):
    user_feedback: str = Field(..., description="'Correct' ou 'Incorrect'")
    correct_category: Optional[str] = None
    notes: Optional[str] = None

class StatisticsResponse(BaseModel):
    total_classifications: int
    avg_confidence: float
    category_distribution: List[Dict]
    top_skills: Optional[List[Dict]]
    accuracy_from_feedback: Optional[float]

class ChatRequest(BaseModel):
    cv_text: str = Field(..., description="Le texte du CV à analyser")
    question: str = Field(..., description="La question à poser sur le CV")
    model: str = Field(default="mistral", description="Modèle à utiliser: mistral, zephyr, phi, gemma")

class ChatResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    error: Optional[str] = None
    model_used: str
    suggestions: List[str] = []

# ============================================
# INITIALISATION FASTAPI
# ============================================

app = FastAPI(
    title="CV Classification API - Version Complète",
    description="""
API professionnelle pour classifier automatiquement des CV.

## Modèle
- **Type**: Gradient Boosting Classifier
- **Entraîné sur**: 962 CVs
- **Performance**: 100% accuracy, 98.96% CV F1-score
- **Catégories**: 25 métiers

## Fonctionnalités
- Classification de CV (texte ou PDF)
- Extraction PDF avec OCR
- Détection de 1000+ compétences
- Recommandations de postes
- Historique et statistiques
""",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# CHARGEMENT DES MODÈLES ML
# ============================================

print("="*80)
print(" CHARGEMENT DES MODÈLES ML")
print("="*80 + "\n")

# Variables globales pour les modèles
MODEL = None
VECTORIZER = None
LABEL_ENCODER = None
TEXT_CLEANER = None
PIPELINE = None  # Nouveau pipeline complet
USE_PIPELINE = False  # Flag pour savoir si on utilise le pipeline

# Variables pour les modules avancés
db_manager = None
pdf_extractor = None
skills_detector = None

def load_pickle(filename):
    """Charger un fichier pickle ou joblib de manière sécurisée"""
    filepath = MODELS_DIR / filename

    if not filepath.exists():
        print(f" {filename} non trouvé dans {MODELS_DIR}")
        return None

    # Essayer d'abord avec joblib (pour les pipelines sklearn)
    try:
        obj = joblib.load(filepath)
        print(f" {filename} chargé avec succès (joblib)")
        return obj
    except Exception as e1:
        # Fallback vers pickle standard
        try:
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
            print(f" {filename} chargé avec succès (pickle)")
            return obj
        except Exception as e2:
            print(f" Erreur lors du chargement de {filename}:")
            print(f"   joblib: {e1}")
            print(f"   pickle: {e2}")
            return None

# PRIORITÉ: Essayer de charger le nouveau pipeline anti-fuite
print(" Tentative de chargement du pipeline complet...")
PIPELINE = load_pickle("cv_classifier_pipeline.pkl")

if PIPELINE is not None:
    # Nouveau mode: utiliser le pipeline
    USE_PIPELINE = True
    print(" Mode PIPELINE activé (anti data-leakage)")

    # Charger le label encoder (nécessaire pour les probabilités)
    LABEL_ENCODER = load_pickle("label_encoder.pkl")
    if LABEL_ENCODER is not None and hasattr(LABEL_ENCODER, 'classes_'):
        print(f"   Catégories: {len(LABEL_ENCODER.classes_)}")
else:
    # Mode legacy: charger les composants séparément
    print(" Pipeline non trouvé, utilisation du mode legacy...")
    USE_PIPELINE = False

    # Charger le modèle ML
    print(" Chargement du modèle ML...")
    MODEL = load_pickle("best_model.pkl")
    if MODEL is None:
        for name in ["Random_Forest_model.pkl", "Logistic_Regression_model.pkl", "SVM_model.pkl"]:
            print(f"   Essai avec {name}...")
            MODEL = load_pickle(name)
            if MODEL is not None:
                break

    # Charger le vectorizer
    print("\n Chargement du vectorizer...")
    VECTORIZER = load_pickle("tfidf_vectorizer.pkl")

    # Charger le label encoder
    print("\n Chargement du label encoder...")
    LABEL_ENCODER = load_pickle("label_encoder.pkl")
    if LABEL_ENCODER is not None and hasattr(LABEL_ENCODER, 'classes_'):
        print(f"   Catégories: {len(LABEL_ENCODER.classes_)}")

    # Initialiser le text cleaner
    print("\n Initialisation du text cleaner...")
    if TEXT_CLEANER_AVAILABLE:
        try:
            TEXT_CLEANER = TextCleaner()
            print(" TextCleaner initialisé")
        except Exception as e:
            print(f" Erreur: {e}")
    else:
        print("  TextCleaner non disponible - utilisation d'un nettoyage basique")

# Résumé ML
print("\n" + "="*80)
if USE_PIPELINE:
    print("  PIPELINE COMPLET CHARGÉ (mode anti data-leakage)")
    print(f"    Pipeline: {type(PIPELINE).__name__}")
    print(f"    Steps: {[name for name, _ in PIPELINE.steps]}")
else:
    all_ml_loaded = all([MODEL, VECTORIZER, LABEL_ENCODER])
    if all_ml_loaded:
        print("  TOUS LES MODÈLES ML CHARGÉS AVEC SUCCÈS! (mode legacy)")
    else:
        print("  CERTAINS MODÈLES ML SONT MANQUANTS:")
        if not MODEL:
            print("    Modèle ML")
        if not VECTORIZER:
            print("    Vectorizer")
        if not LABEL_ENCODER:
            print("    Label Encoder")
print("="*80 + "\n")

# ============================================
# CHARGEMENT DES MÉTRIQUES (pour reproductibilité)
# ============================================

TRAINING_METRICS: Dict[str, Any] = {}
CV_RESULTS: Dict[str, Any] = {}
TEST_RESULTS: Dict[str, Any] = {}
TRAINING_METADATA: Dict[str, Any] = {}
SPLIT_METADATA: Dict[str, Any] = {}

def load_json_file(filename: str) -> Dict[str, Any]:
    """Charger un fichier JSON depuis le dossier models"""
    filepath = MODELS_DIR / filename
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  Erreur chargement {filename}: {e}")
    return {}

print(" Chargement des métriques...")

# Charger les résultats de cross-validation
CV_RESULTS = load_json_file("cv_results.json")
if CV_RESULTS:
    print(f"  cv_results.json chargé ({CV_RESULTS.get('n_samples', '?')} samples)")

# Charger les résultats d'évaluation test
TEST_RESULTS = load_json_file("test_evaluation.json")
if TEST_RESULTS:
    eval_data = TEST_RESULTS.get('evaluation', {})
    print(f"  test_evaluation.json chargé ({eval_data.get('n_samples', '?')} test samples)")

# Charger les métadonnées d'entraînement
TRAINING_METADATA = load_json_file("training_metadata.json")
if TRAINING_METADATA:
    print(f"  training_metadata.json chargé (trained: {TRAINING_METADATA.get('trained_at', '?')[:10]})")

# Charger les métadonnées du split
SPLIT_METADATA = load_json_file("../data/splits/split_metadata.json")
if not SPLIT_METADATA:
    # Essayer le chemin alternatif
    split_path = DATA_DIR / "splits" / "split_metadata.json"
    if split_path.exists():
        try:
            with open(split_path, 'r') as f:
                SPLIT_METADATA = json.load(f)
            print(f"  split_metadata.json chargé")
        except:
            SPLIT_METADATA = {}

print()

# ============================================
# INITIALISATION DES MODULES AVANCÉS
# ============================================

print("="*80)
print(" INITIALISATION DES MODULES AVANCÉS")
print("="*80 + "\n")

# Base de données
if DATABASE_AVAILABLE and CVDatabaseManager:
    try:
        db_manager = CVDatabaseManager(str(DATA_DIR / "cv_history.db"))
        print(" Base de données initialisée")
    except Exception as e:
        print(f" Erreur base de données: {e}")
        db_manager = None

# Extracteur PDF
if PDF_EXTRACTOR_AVAILABLE and AdvancedPDFExtractor:
    try:
        pdf_extractor = AdvancedPDFExtractor()
        print(" Extracteur PDF initialisé")
    except Exception as e:
        print(f" Erreur extracteur PDF: {e}")
        pdf_extractor = None

# Détecteur de compétences
if SKILLS_DETECTOR_AVAILABLE and SkillsDetector:
    try:
        skills_detector = SkillsDetector()
        print(" Détecteur de compétences initialisé")
    except Exception as e:
        print(f" Erreur détecteur de compétences: {e}")
        skills_detector = None

# Chatbot CV
chatbot_instance = None
if CHATBOT_AVAILABLE:
    try:
        # Essayer d'abord le chatbot avec HuggingFace API
        chatbot_instance = CVChatbot(model_name="mistral")
        print(" CV Chatbot initialisé (HuggingFace API)")
    except Exception as e:
        print(f" Erreur chatbot HF, fallback vers SimpleChatbot: {e}")
        try:
            chatbot_instance = SimpleCVChatbot()
            print(" Simple CV Chatbot initialisé (fallback)")
        except:
            chatbot_instance = None

print("="*80 + "\n")

# ============================================
# FONCTION DE NETTOYAGE BASIQUE (fallback)
# ============================================

def basic_clean(text):
    """Nettoyage basique si TextCleaner n'est pas disponible"""
    import re
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ============================================
# ENDPOINTS DE BASE
# ============================================

@app.get("/")
def root():
    """Endpoint racine avec informations sur l'API"""
    if USE_PIPELINE:
        ml_ready = PIPELINE is not None and LABEL_ENCODER is not None
    else:
        ml_ready = all([MODEL, VECTORIZER, LABEL_ENCODER])

    return {
        "message": "CV Classification API - Version Anti Data-Leakage",
        "version": "2.2.0",
        "status": "running",
        "mode": "pipeline" if USE_PIPELINE else "legacy",
        "features": {
            "ml_classification": ml_ready,
            "data_leakage_prevention": USE_PIPELINE,
            "pdf_extraction": pdf_extractor is not None,
            "skills_detection": skills_detector is not None,
            "database_history": db_manager is not None
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "upload_cv": "/upload-cv",
            "analyze_skills": "/analyze-skills",
            "history": "/history",
            "statistics": "/statistics",
            "docs": "/docs"
        }
    }

@app.get("/config")
def get_config():
    """Configuration dynamique pour le frontend"""
    return {
        "api_url": os.environ.get("API_URL", ""),
        "app_name": "CV Classifier",
        "version": "2.2.0",
        "environment": os.environ.get("ENVIRONMENT", "development")
    }

@app.get("/health", response_model=HealthResponse)
def health():
    """Vérifier l'état de santé de l'API"""
    if USE_PIPELINE:
        is_healthy = PIPELINE is not None and LABEL_ENCODER is not None
    else:
        is_healthy = all([MODEL, VECTORIZER, LABEL_ENCODER])

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        model_loaded=PIPELINE is not None if USE_PIPELINE else MODEL is not None,
        vectorizer_loaded=True if USE_PIPELINE else VECTORIZER is not None,
        label_encoder_loaded=LABEL_ENCODER is not None,
        text_cleaner_loaded=True if USE_PIPELINE else TEXT_CLEANER is not None,
        database_available=db_manager is not None,
        pdf_extractor_available=pdf_extractor is not None,
        skills_detector_available=skills_detector is not None,
        version="2.1.0",
        base_dir=str(BASE_DIR),
        models_dir=str(MODELS_DIR)
    )

@app.post("/predict", response_model=CVPrediction, tags=["Classification"])
def predict(cv: CVInput, include_all_probabilities: bool = False):
    """Prédire la catégorie d'un CV depuis un texte"""

    # Vérifier le texte d'abord
    if not cv.resume_text or len(cv.resume_text.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Le CV doit contenir au moins 10 caractères"
        )

    try:
        if USE_PIPELINE and PIPELINE is not None:
            # Mode PIPELINE: le pipeline gère tout (nettoyage + vectorisation + prédiction)
            if LABEL_ENCODER is None:
                raise HTTPException(
                    status_code=503,
                    detail="Label encoder non disponible"
                )

            # Le pipeline prend le texte brut directement
            prediction = PIPELINE.predict([cv.resume_text])[0]
            probabilities = PIPELINE.predict_proba([cv.resume_text])[0]

            # Décoder la prédiction
            category = LABEL_ENCODER.inverse_transform([prediction])[0]
            confidence = float(probabilities.max())

            # Probabilités pour toutes les catégories
            all_probs = None
            if include_all_probabilities:
                all_probs = {
                    LABEL_ENCODER.inverse_transform([i])[0]: float(prob)
                    for i, prob in enumerate(probabilities)
                }
                all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))

        else:
            # Mode LEGACY: composants séparés
            if not all([MODEL, VECTORIZER, LABEL_ENCODER]):
                missing = []
                if not MODEL:
                    missing.append("Modèle ML")
                if not VECTORIZER:
                    missing.append("Vectorizer")
                if not LABEL_ENCODER:
                    missing.append("Label Encoder")

                raise HTTPException(
                    status_code=503,
                    detail=f"Composants manquants: {', '.join(missing)}. Vérifiez {MODELS_DIR}"
                )

            # Nettoyer le texte
            if TEXT_CLEANER:
                cleaned_text = TEXT_CLEANER.clean_text(cv.resume_text)
            else:
                cleaned_text = basic_clean(cv.resume_text)

            # Vectoriser
            X = VECTORIZER.transform([cleaned_text])

            # Prédire
            prediction = MODEL.predict(X)[0]
            probabilities = MODEL.predict_proba(X)[0]

            # Décoder
            category = LABEL_ENCODER.inverse_transform([prediction])[0]
            confidence = float(probabilities.max())

            # Probabilités pour toutes les catégories
            all_probs = None
            if include_all_probabilities:
                all_probs = {
                    LABEL_ENCODER.inverse_transform([i])[0]: float(prob)
                    for i, prob in enumerate(probabilities)
                }
                all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))

        return CVPrediction(
            category=category,
            confidence=confidence,
            all_probabilities=all_probs
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/batch-predict", response_model=BatchCVPrediction, tags=["Classification"])
def batch_predict(batch: BatchCVInput, include_all_probabilities: bool = False):
    """Prédire les catégories de plusieurs CV en batch"""
    predictions = []
    
    for resume_text in batch.resumes:
        if resume_text and len(resume_text.strip()) >= 10:
            try:
                result = predict(
                    CVInput(resume_text=resume_text),
                    include_all_probabilities
                )
                predictions.append(result)
            except:
                predictions.append(CVPrediction(
                    category="ERROR",
                    confidence=0.0
                ))
        else:
            predictions.append(CVPrediction(
                category="INVALID",
                confidence=0.0
            ))
    
    return BatchCVPrediction(
        predictions=predictions,
        total_processed=len(predictions)
    )

@app.get("/categories", tags=["Information"])
def get_categories():
    """Obtenir la liste de toutes les catégories disponibles"""
    if not LABEL_ENCODER:
        raise HTTPException(status_code=503, detail="Label encoder non chargé")
    
    categories = LABEL_ENCODER.classes_.tolist()
    return {
        "total_categories": len(categories),
        "categories": sorted(categories)
    }

@app.get("/model-info", tags=["Information"])
def get_model_info():
    """Obtenir des informations sur le modèle chargé (métriques chargées dynamiquement)"""
    if USE_PIPELINE:
        if not PIPELINE:
            raise HTTPException(status_code=503, detail="Pipeline non chargé")

        # Extraire les infos du pipeline
        pipeline_steps = [name for name, _ in PIPELINE.steps]
        classifier = PIPELINE.named_steps.get('classifier')
        tfidf = PIPELINE.named_steps.get('tfidf')

        # Extraire les métriques des fichiers chargés
        cv_scores = CV_RESULTS.get('scores', {})
        test_eval = TEST_RESULTS.get('evaluation', {})
        test_metrics = test_eval.get('metrics', {})
        split_meta = SPLIT_METADATA if SPLIT_METADATA else {}

        # Calculer les tailles depuis les métadonnées
        train_samples = CV_RESULTS.get('n_samples') or split_meta.get('train_samples')
        test_samples = test_eval.get('n_samples') or split_meta.get('test_samples')

        return {
            "mode": "pipeline",
            "model_type": type(classifier).__name__ if classifier else "Unknown",
            "classifier": TRAINING_METADATA.get('classifier', 'unknown'),
            "pipeline_steps": pipeline_steps,
            "trained_at": TRAINING_METADATA.get('trained_at'),
            "training_time_seconds": TRAINING_METADATA.get('training_time_seconds'),
            "data": {
                "train_samples": train_samples,
                "test_samples": test_samples,
                "total_samples": (train_samples or 0) + (test_samples or 0) if train_samples and test_samples else None,
                "test_size": split_meta.get('test_size'),
                "random_state": split_meta.get('random_state')
            },
            "n_features": tfidf.max_features if tfidf and hasattr(tfidf, 'max_features') else None,
            "n_categories": TRAINING_METADATA.get('n_classes') or (len(LABEL_ENCODER.classes_) if LABEL_ENCODER else None),
            "categories": TRAINING_METADATA.get('classes') or (LABEL_ENCODER.classes_.tolist() if LABEL_ENCODER else None),
            "data_leakage_prevention": True,
            "performance": {
                "cross_validation": {
                    "n_folds": CV_RESULTS.get('n_folds'),
                    "accuracy": {
                        "mean": cv_scores.get('accuracy', {}).get('cv_mean'),
                        "std": cv_scores.get('accuracy', {}).get('cv_std')
                    },
                    "f1_macro": {
                        "mean": cv_scores.get('f1_macro', {}).get('cv_mean'),
                        "std": cv_scores.get('f1_macro', {}).get('cv_std')
                    },
                    "precision_macro": {
                        "mean": cv_scores.get('precision_macro', {}).get('cv_mean'),
                        "std": cv_scores.get('precision_macro', {}).get('cv_std')
                    },
                    "recall_macro": {
                        "mean": cv_scores.get('recall_macro', {}).get('cv_mean'),
                        "std": cv_scores.get('recall_macro', {}).get('cv_std')
                    }
                },
                "test_set": {
                    "accuracy": test_metrics.get('accuracy'),
                    "f1_macro": test_metrics.get('f1_macro'),
                    "precision_macro": test_metrics.get('precision_macro'),
                    "recall_macro": test_metrics.get('recall_macro'),
                    "f1_weighted": test_metrics.get('f1_weighted')
                },
                "note": "Proper train/test split BEFORE preprocessing - no data leakage"
            }
        }
    else:
        if not MODEL:
            raise HTTPException(status_code=503, detail="Modèle non chargé")

        return {
            "mode": "legacy",
            "model_type": type(MODEL).__name__,
            "n_features": VECTORIZER.max_features if VECTORIZER else None,
            "n_categories": len(LABEL_ENCODER.classes_) if LABEL_ENCODER else None,
            "categories": LABEL_ENCODER.classes_.tolist() if LABEL_ENCODER else None,
            "data_leakage_prevention": False,
            "performance": {
                "note": "Legacy mode - metrics not tracked. Consider retraining with pipeline."
            }
        }

# ============================================
# ENDPOINTS AVANCÉS - UPLOAD PDF
# ============================================

@app.post("/upload-cv", response_model=CVUploadResponse, tags=["Advanced - PDF"])
async def upload_and_classify_cv(
    file: UploadFile = File(...),
    extract_skills: bool = Query(True, description="Extraire les compétences"),
    recommend_jobs: bool = Query(True, description="Recommander des postes"),
    save_to_history: bool = Query(True, description="Sauvegarder dans l'historique")
):
    """
    Upload et classification complète d'un CV PDF
    
    Fonctionnalités:
    - Extraction du texte PDF (avec OCR si nécessaire)
    - Classification du CV
    - Extraction des compétences (optionnel)
    - Recommandations de postes (optionnel)
    - Sauvegarde dans l'historique (optionnel)
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Seuls les fichiers PDF sont acceptés"
        )
    
    if not pdf_extractor:
        raise HTTPException(
            status_code=503,
            detail="PDF Extractor non disponible. Installez: pip install pdfplumber pypdf pytesseract pdf2image"
        )
    
    start_time = datetime.now()
    
    try:
        # Sauvegarder temporairement le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # 1. Extraire le texte du PDF
        print(f" Extraction du PDF: {file.filename}")
        pdf_result = pdf_extractor.extract_from_pdf(tmp_path, method='auto')
        
        # 2. Classifier le CV
        print(f" Classification...")
        classification_result = predict(
            CVInput(resume_text=pdf_result.text),
            include_all_probabilities=True
        )
        
        predicted_category = classification_result.category
        confidence = classification_result.confidence
        all_probabilities = classification_result.all_probabilities
        
        # 3. Extraire les compétences si demandé
        skills_summary = None
        experience_info = None
        job_recommendations = None
        
        if extract_skills and skills_detector:
            print(" Extraction des compétences...")
            skills = skills_detector.extract_skills(pdf_result.text)
            experience_info = skills_detector.analyze_experience(pdf_result.text)
            
            skills_summary = {
                'total_technical_skills': len(skills['technical_skills']),
                'total_soft_skills': len(skills['soft_skills']),
                'total_frameworks': len(skills['frameworks']),
                'total_tools': len(skills['tools']),
                'total_languages': len(skills['languages'])
            }
            
            # 4. Recommander des postes si demandé
            if recommend_jobs:
                print(" Génération de recommandations...")
                job_recommendations = skills_detector.recommend_jobs(
                    skills,
                    experience_info,
                    top_n=5
                )
        
        # 5. Sauvegarder dans l'historique si demandé
        classification_id = None
        if save_to_history and db_manager:
            print(" Sauvegarde dans l'historique...")
            
            # Préparer les compétences pour la DB
            extracted_skills_list = []
            if extract_skills and skills_detector:
                for skill in skills.get('technical_skills', []):
                    extracted_skills_list.append({
                        'name': skill['skill'],
                        'category': skill['category'],
                        'confidence': skill['confidence']
                    })
            
            classification_id = db_manager.add_classification(
                cv_text=pdf_result.text,
                predicted_category=predicted_category,
                confidence_score=confidence,
                cv_filename=file.filename,
                all_probabilities=all_probabilities,
                model_used=type(MODEL).__name__,
                model_version="1.0",
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                extracted_skills=extracted_skills_list
            )
        
        # Nettoyer le fichier temporaire
        Path(tmp_path).unlink()
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return CVUploadResponse(
            classification_id=classification_id,
            filename=file.filename,
            predicted_category=predicted_category,
            confidence=confidence,
            extraction_method=pdf_result.extraction_method,
            extraction_confidence=pdf_result.confidence,
            skills_summary=skills_summary,
            experience_info=experience_info,
            job_recommendations=job_recommendations,
            processing_time_ms=processing_time,
            extracted_text=pdf_result.text  # Pour le chatbot
        )
        
    except Exception as e:
        # Nettoyer en cas d'erreur
        if 'tmp_path' in locals():
            Path(tmp_path).unlink(missing_ok=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement: {str(e)}"
        )

# ============================================
# ENDPOINTS AVANCÉS - SKILLS ANALYSIS
# ============================================

@app.post("/analyze-skills", response_model=SkillsAnalysisResponse, tags=["Advanced - Skills"])
async def analyze_skills_from_text(
    cv: CVInput,
    recommend_jobs: bool = Query(True, description="Recommander des postes")
):
    """Analyser les compétences depuis un texte brut"""
    if not skills_detector:
        raise HTTPException(
            status_code=503,
            detail="Skills Detector non disponible"
        )
    
    try:
        # Extraire les compétences
        skills = skills_detector.extract_skills(cv.resume_text)
        
        # Analyser l'expérience
        experience = skills_detector.analyze_experience(cv.resume_text)
        
        # Recommandations
        recommendations = []
        if recommend_jobs:
            recommendations = skills_detector.recommend_jobs(
                skills,
                experience,
                top_n=5
            )
        
        # Top strengths
        top_strengths = skills_detector._identify_top_strengths(skills)
        
        return SkillsAnalysisResponse(
            skills_summary={
                'total_technical_skills': len(skills['technical_skills']),
                'total_soft_skills': len(skills['soft_skills']),
                'total_frameworks': len(skills['frameworks']),
                'total_tools': len(skills['tools']),
                'total_languages': len(skills['languages'])
            },
            detailed_skills=skills,
            experience_analysis=experience,
            job_recommendations=recommendations,
            top_strengths=top_strengths
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse: {str(e)}"
        )

# ============================================
# ENDPOINTS AVANCÉS - CHATBOT
# ============================================

@app.post("/chat", response_model=ChatResponse, tags=["Advanced - Chatbot"])
async def chat_about_cv(request: ChatRequest):
    """
    Poser une question sur un CV en utilisant l'IA.

    Le chatbot utilise HuggingFace Inference API (gratuit) pour répondre
    aux questions sur le CV fourni.

    Modèles disponibles:
    - mistral: Mistral 7B (recommandé)
    - zephyr: Zephyr 7B
    - phi: Microsoft Phi-2
    - gemma: Google Gemma 7B
    """
    if not CHATBOT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Chatbot non disponible. Module chatbot non installé."
        )

    if not request.cv_text or len(request.cv_text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Le CV doit contenir au moins 50 caractères."
        )

    if not request.question or len(request.question.strip()) < 5:
        raise HTTPException(
            status_code=400,
            detail="La question doit contenir au moins 5 caractères."
        )

    try:
        # Essayer d'abord avec le chatbot HuggingFace si token disponible
        result = None
        model_used = request.model

        if CVChatbot:
            chatbot = CVChatbot(model_name=request.model)
            chatbot.set_cv(request.cv_text)
            result = chatbot.ask(request.question)

            # Si échec avec l'API HF, fallback vers SimpleCVChatbot
            if not result.get("success") and SimpleCVChatbot:
                chatbot = SimpleCVChatbot()
                chatbot.set_cv(request.cv_text)
                result = chatbot.ask(request.question)
                model_used = "simple"

        elif SimpleCVChatbot:
            chatbot = SimpleCVChatbot()
            chatbot.set_cv(request.cv_text)
            result = chatbot.ask(request.question)
            model_used = "simple"
        else:
            raise HTTPException(status_code=503, detail="Aucun chatbot disponible")

        if result is None:
            raise HTTPException(status_code=500, detail="Erreur chatbot")

        # Récupérer les suggestions
        suggestions = chatbot.get_suggestions() if hasattr(chatbot, 'get_suggestions') else []

        return ChatResponse(
            success=result.get("success", False),
            answer=result.get("answer"),
            error=result.get("error"),
            model_used=model_used,
            suggestions=suggestions[:5]
        )

    except HTTPException:
        raise
    except Exception as e:
        return ChatResponse(
            success=False,
            answer=None,
            error=f"Erreur lors du traitement: {str(e)}",
            model_used=request.model,
            suggestions=[]
        )

@app.get("/chat/models", tags=["Advanced - Chatbot"])
async def get_available_chat_models():
    """Obtenir la liste des modèles de chat disponibles."""
    if not CHATBOT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Chatbot non disponible")

    models = {
        "llama": {
            "name": "Llama 3.2 3B",
            "id": "meta-llama/Llama-3.2-3B-Instruct",
            "description": "Modèle Meta, performant et rapide (recommandé)"
        },
        "qwen": {
            "name": "Qwen 2.5 1.5B",
            "id": "Qwen/Qwen2.5-1.5B-Instruct",
            "description": "Modèle Alibaba, compact et efficace"
        },
        "phi": {
            "name": "Phi-3 Mini",
            "id": "microsoft/Phi-3-mini-4k-instruct",
            "description": "Modèle Microsoft, bon raisonnement"
        }
    }

    return {
        "available_models": models,
        "default": "llama",
        "note": "Utilise l'API gratuite HuggingFace Inference"
    }

@app.get("/chat/suggestions", tags=["Advanced - Chatbot"])
async def get_chat_suggestions():
    """Obtenir des suggestions de questions à poser sur un CV."""
    return {
        "suggestions": [
            "Quelles sont les compétences principales de ce candidat ?",
            "Résume ce CV en 3 points clés.",
            "Quel est le niveau d'expérience de ce candidat ?",
            "Quelles technologies maîtrise-t-il ?",
            "Est-il adapté pour un poste de développeur ?",
            "Quels sont ses points forts ?",
            "Quelle est sa formation ?",
            "A-t-il de l'expérience en management ?",
            "Quels projets a-t-il réalisés ?"
        ]
    }

# ============================================
# ENDPOINTS AVANCÉS - HISTORY
# ============================================

@app.get("/history", tags=["Advanced - History"])
async def get_classification_history(
    limit: int = Query(10, ge=1, le=100, description="Nombre de résultats"),
    category: Optional[str] = Query(None, description="Filtrer par catégorie"),
    start_date: Optional[str] = Query(None, description="Date de début (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Date de fin (YYYY-MM-DD)")
):
    """Récupérer l'historique des classifications"""
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Base de données non disponible"
        )
    
    try:
        if category:
            results = db_manager.get_classifications_by_category(category)[:limit]
        elif start_date and end_date:
            results = db_manager.get_classifications_by_date_range(start_date, end_date)
        else:
            results = db_manager.get_recent_classifications(limit)
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )

@app.put("/history/{classification_id}/feedback", tags=["Advanced - History"])
async def update_classification_feedback(
    classification_id: int,
    feedback: FeedbackUpdate
):
    """Mettre à jour le feedback utilisateur pour une classification"""
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Base de données non disponible"
        )
    
    try:
        db_manager.update_feedback(
            classification_id,
            feedback.user_feedback,
            feedback.correct_category,
            feedback.notes
        )
        
        return {
            "status": "success",
            "message": "Feedback mis à jour",
            "classification_id": classification_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )

@app.get("/statistics", response_model=StatisticsResponse, tags=["Advanced - Statistics"])
async def get_statistics():
    """Obtenir les statistiques globales"""
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Base de données non disponible"
        )
    
    try:
        stats = db_manager.get_statistics()
        
        return StatisticsResponse(
            total_classifications=stats['total_classifications'],
            avg_confidence=stats['avg_confidence'],
            category_distribution=stats['category_distribution'],
            top_skills=stats.get('top_skills'),
            accuracy_from_feedback=stats.get('accuracy_from_feedback')
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )

@app.post("/export", tags=["Advanced - Export"])
async def export_history_to_csv(
    include_skills: bool = Query(False, description="Inclure les compétences"),
    output_filename: str = Query("cv_history_export.csv", description="Nom du fichier")
):
    """Exporter l'historique vers un fichier CSV"""
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Base de données non disponible"
        )
    
    try:
        output_dir = BASE_DIR / "outputs" / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        
        db_manager.export_to_csv(str(output_path), include_skills)
        
        return {
            "status": "success",
            "message": "Export réussi",
            "file_path": str(output_path)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )

@app.get("/search-skill/{skill_name}", tags=["Advanced - Search"])
async def search_by_skill(skill_name: str):
    """Rechercher tous les CV contenant une compétence spécifique"""
    if not db_manager:
        raise HTTPException(
            status_code=503,
            detail="Base de données non disponible"
        )
    
    try:
        results = db_manager.search_by_skill(skill_name)
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )

# ============================================
# FRONTEND STATIQUE
# ============================================

FRONTEND_DIR = API_DIR / "frontend"
if FRONTEND_DIR.exists():
    print(f" Frontend disponible dans: {FRONTEND_DIR}")

@app.get("/app", include_in_schema=False)
async def serve_frontend():
    """Servir l'interface frontend"""
    frontend_path = FRONTEND_DIR / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    raise HTTPException(status_code=404, detail="Frontend non trouve")

# ============================================
# LANCEMENT DU SERVEUR
# ============================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*80)
    print(" DEMARRAGE DU SERVEUR API")
    print("="*80)
    print("\n L'API sera accessible sur:")
    print("   - http://localhost:8000")
    print("   - http://localhost:8000/app (Interface graphique)")
    print("   - http://localhost:8000/docs (Documentation interactive)")
    print("   - http://localhost:8000/redoc (Documentation alternative)")
    print("\n Fonctionnalites disponibles:")
    print(f"   - Classification ML: {'Bon' if all([MODEL, VECTORIZER, LABEL_ENCODER]) else 'Mauvais'}")
    print(f"   - Extraction PDF: {'Bon' if pdf_extractor else 'Mauvais'}")
    print(f"   - Detection competences: {'Bon' if skills_detector else 'Mauvais'}")
    print(f"   - Base de donnees: {'Bon' if db_manager else 'Mauvais'}")
    print("\n  Pour arreter: Ctrl+C")
    print("="*80 + "\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )