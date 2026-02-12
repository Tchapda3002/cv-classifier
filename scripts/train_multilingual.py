#!/usr/bin/env python3
"""
Training script avec Sentence Transformers (Multilingue)
Remplace TF-IDF par des embeddings multilingues
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Ajouter le chemin du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

print("="*70)
print(" TRAINING MULTILINGUE - Sentence Transformers + Random Forest")
print("="*70)

# 1. Charger le modèle Sentence Transformer
print("\n[1/6] Chargement du modèle multilingue...")
start_load = time.time()

from sentence_transformers import SentenceTransformer

# Modèle multilingue léger (50+ langues)
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
embedder = SentenceTransformer(MODEL_NAME)
print(f"      Modèle: {MODEL_NAME}")
print(f"      Dimension embeddings: {embedder.get_sentence_embedding_dimension()}")
print(f"      Temps de chargement: {time.time() - start_load:.2f}s")

# 2. Charger les données
print("\n[2/6] Chargement des données...")
data_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'resume_dataset.csv')
df = pd.read_csv(data_path)
print(f"      Samples: {len(df)}")
print(f"      Catégories: {df['Category'].nunique()}")

# Utiliser le texte brut (pas besoin de cleaning avec les transformers)
X_raw = df['Resume'].values
y = df['Category'].values

# 3. Encoder les labels
print("\n[3/6] Encodage des labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"      Classes: {list(label_encoder.classes_)}")

# 4. Split des données
print("\n[4/6] Split train/test...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(f"      Train: {len(X_train_raw)} samples")
print(f"      Test: {len(X_test_raw)} samples")

# 5. Générer les embeddings
print("\n[5/6] Génération des embeddings...")
print("      (Cela peut prendre quelques minutes...)")

start_embed = time.time()
print("      Encoding train set...")
X_train = embedder.encode(X_train_raw.tolist(), show_progress_bar=True)
print("      Encoding test set...")
X_test = embedder.encode(X_test_raw.tolist(), show_progress_bar=True)
embed_time = time.time() - start_embed

print(f"      Shape train: {X_train.shape}")
print(f"      Shape test: {X_test.shape}")
print(f"      Temps d'encodage: {embed_time:.2f}s")

# 6. Entraîner le Random Forest
print("\n[6/6] Entraînement du Random Forest...")
start_train = time.time()

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Cross-validation sur train
print("      Cross-validation 5-fold...")
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')
print(f"      CV F1-scores: {cv_scores}")
print(f"      CV F1 moyen: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Entraînement final
print("      Entraînement final...")
clf.fit(X_train, y_train)
train_time = time.time() - start_train
print(f"      Temps d'entraînement: {train_time:.2f}s")

# 7. Évaluation
print("\n" + "="*70)
print(" ÉVALUATION SUR LE TEST SET")
print("="*70)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n   Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"   F1-Score:  {f1:.4f} ({f1*100:.1f}%)")

print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 8. Test multilingue
print("\n" + "="*70)
print(" TEST MULTILINGUE")
print("="*70)

test_cvs = [
    ("Data Scientist with Python, TensorFlow, Machine Learning experience", "EN"),
    ("Scientifique des données avec Python, TensorFlow, Machine Learning", "FR"),
    ("Java Developer with Spring Boot, Hibernate, Microservices", "EN"),
    ("Développeur Java avec Spring Boot, Hibernate, Microservices", "FR"),
    ("HR Manager recruitment training payroll employee relations", "EN"),
    ("Responsable RH recrutement formation paie relations employés", "FR"),
    ("Sales Manager business development negotiation revenue", "EN"),
    ("Commercial développement business négociation chiffre affaires vente", "FR"),
]

print("\n   Prédictions multilingues:\n")
for cv_text, lang in test_cvs:
    embedding = embedder.encode([cv_text])
    pred = clf.predict(embedding)[0]
    proba = clf.predict_proba(embedding).max()
    category = label_encoder.inverse_transform([pred])[0]
    print(f"   [{lang}] {cv_text[:50]}...")
    print(f"        → {category} (confiance: {proba:.1%})\n")

# 9. Sauvegarder les modèles
print("\n" + "="*70)
print(" SAUVEGARDE DES MODÈLES")
print("="*70)

models_dir = os.path.join(PROJECT_ROOT, 'models', 'multilingual')
os.makedirs(models_dir, exist_ok=True)

# Sauvegarder le classifieur
clf_path = os.path.join(models_dir, 'classifier_multilingual.pkl')
joblib.dump(clf, clf_path)
print(f"   Classifieur: {clf_path}")

# Sauvegarder le label encoder
le_path = os.path.join(models_dir, 'label_encoder_multilingual.pkl')
joblib.dump(label_encoder, le_path)
print(f"   Label Encoder: {le_path}")

# Sauvegarder les métadonnées
metadata = {
    "model_name": MODEL_NAME,
    "embedding_dim": embedder.get_sentence_embedding_dimension(),
    "n_classes": len(label_encoder.classes_),
    "classes": list(label_encoder.classes_),
    "accuracy": float(accuracy),
    "f1_score": float(f1),
    "cv_f1_mean": float(cv_scores.mean()),
    "cv_f1_std": float(cv_scores.std()),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "trained_at": datetime.now().isoformat(),
    "training_time_seconds": train_time,
    "embedding_time_seconds": embed_time
}

meta_path = os.path.join(models_dir, 'metadata.json')
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   Métadonnées: {meta_path}")

print("\n" + "="*70)
print(" RÉSUMÉ")
print("="*70)
print(f"""
   Modèle:           {MODEL_NAME}
   Accuracy:         {accuracy*100:.1f}%
   F1-Score:         {f1*100:.1f}%
   CV F1 moyen:      {cv_scores.mean()*100:.1f}%

   Temps total:      {embed_time + train_time:.1f}s

   Multilingue:      ✓ Oui (50+ langues)
""")
print("="*70)
