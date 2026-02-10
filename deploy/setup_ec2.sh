#!/bin/bash
# ==============================================
# Script de configuration EC2 pour CV Classifier
# Compatible AWS Free Tier (t2.micro)
# ==============================================

set -e

echo "=========================================="
echo " Configuration EC2 - CV Classifier"
echo "=========================================="

# Variables
APP_DIR="/home/ubuntu/cv-classifier"
VENV_DIR="$APP_DIR/venv"

# 1. Mise à jour système
echo "[1/7] Mise à jour du système..."
sudo apt update && sudo apt upgrade -y

# 2. Installation des dépendances système
echo "[2/7] Installation des dépendances..."
sudo apt install -y python3-pip python3-venv git

# 3. Création du swap (important pour t2.micro avec 1GB RAM)
echo "[3/7] Configuration du swap (2GB)..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "Swap créé avec succès"
else
    echo "Swap déjà configuré"
fi

# 4. Cloner ou mettre à jour le projet
echo "[4/7] Configuration du projet..."
if [ -d "$APP_DIR" ]; then
    echo "Mise à jour du projet existant..."
    cd $APP_DIR
    git pull origin main || true
else
    echo "Clonage du projet..."
    # Remplacer par votre URL de repo
    git clone https://github.com/VOTRE_USERNAME/Projet_NLPfinal.git $APP_DIR
    cd $APP_DIR
fi

# 5. Environnement virtuel Python
echo "[5/7] Configuration de l'environnement Python..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi
source $VENV_DIR/bin/activate

# 6. Installation des dépendances Python
echo "[6/7] Installation des dépendances Python..."
pip install --upgrade pip
pip install -r requirements-prod.txt

# Télécharger les données NLTK nécessaires
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

# 7. Configuration du service systemd
echo "[7/7] Configuration du service systemd..."
sudo tee /etc/systemd/system/cv-classifier.service > /dev/null <<EOF
[Unit]
Description=CV Classifier API
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV_DIR/bin"
EnvironmentFile=$APP_DIR/.env
ExecStart=$VENV_DIR/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable cv-classifier

echo ""
echo "=========================================="
echo " Configuration terminée!"
echo "=========================================="
echo ""
echo "Prochaines étapes:"
echo "1. Créer le fichier .env avec votre token HuggingFace:"
echo "   echo 'HF_TOKEN=votre_token' > $APP_DIR/.env"
echo ""
echo "2. Démarrer le service:"
echo "   sudo systemctl start cv-classifier"
echo ""
echo "3. Vérifier le statut:"
echo "   sudo systemctl status cv-classifier"
echo ""
echo "4. Voir les logs:"
echo "   sudo journalctl -u cv-classifier -f"
echo ""
echo "L'API sera accessible sur: http://VOTRE_IP:8000"
echo "Interface web: http://VOTRE_IP:8000/app"
echo ""
