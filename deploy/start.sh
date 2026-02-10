#!/bin/bash
# Script de démarrage rapide pour CV Classifier

cd /home/ubuntu/cv-classifier
source venv/bin/activate

echo "Démarrage de CV Classifier..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
