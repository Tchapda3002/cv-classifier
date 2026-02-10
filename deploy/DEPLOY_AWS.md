# Déploiement AWS EC2 - CV Classifier

## Prérequis

- Compte AWS (Free Tier OK)
- Votre token HuggingFace (obtenir sur https://huggingface.co/settings/tokens)

---

## Étape 1: Créer l'instance EC2

1. Aller sur **AWS Console** → **EC2** → **Launch Instance**

2. Configuration:
   ```
   Nom: cv-classifier
   AMI: Ubuntu Server 22.04 LTS (Free tier eligible)
   Type: t2.micro (Free tier - 1 vCPU, 1GB RAM)
   Key pair: Créer ou sélectionner une clé existante
   ```

3. **Network settings** → Edit:
   ```
   ✅ Allow SSH traffic (port 22)
   ✅ Allow HTTP traffic (port 80)
   ✅ Add rule: Custom TCP, Port 8000, Source: 0.0.0.0/0
   ```

4. **Storage**: 20 GB gp2 (Free tier: 30GB max)

5. Cliquer **Launch instance**

---

## Étape 2: Se connecter à l'instance

```bash
# Télécharger votre clé .pem depuis AWS
chmod 400 votre-cle.pem

# Se connecter (remplacer par votre IP publique)
ssh -i votre-cle.pem ubuntu@XX.XX.XX.XX
```

---

## Étape 3: Uploader le projet

### Option A: Via Git (recommandé)

D'abord, pusher votre projet sur GitHub:
```bash
# Sur votre machine locale
cd /Users/Apple/Desktop/Projets/Projet_NLPfinal
git add .
git commit -m "Prepare for AWS deployment"
git push origin main
```

Puis sur EC2:
```bash
git clone https://github.com/VOTRE_USERNAME/Projet_NLPfinal.git ~/cv-classifier
```

### Option B: Via SCP (upload direct)

```bash
# Depuis votre machine locale
scp -i votre-cle.pem -r /Users/Apple/Desktop/Projets/Projet_NLPfinal ubuntu@XX.XX.XX.XX:~/cv-classifier
```

---

## Étape 4: Exécuter le script de setup

```bash
# Sur EC2
cd ~/cv-classifier
chmod +x deploy/setup_ec2.sh
./deploy/setup_ec2.sh
```

---

## Étape 5: Configurer le token HuggingFace

```bash
echo 'HF_TOKEN=your_huggingface_token_here' > ~/cv-classifier/.env
```

---

## Étape 6: Démarrer l'application

```bash
sudo systemctl start cv-classifier
sudo systemctl status cv-classifier
```

---

## Accéder à l'application

- **API**: `http://VOTRE_IP_PUBLIQUE:8000`
- **Interface**: `http://VOTRE_IP_PUBLIQUE:8000/app`
- **Documentation**: `http://VOTRE_IP_PUBLIQUE:8000/docs`

---

## Commandes utiles

```bash
# Voir les logs
sudo journalctl -u cv-classifier -f

# Redémarrer
sudo systemctl restart cv-classifier

# Arrêter
sudo systemctl stop cv-classifier

# Vérifier la RAM
free -h

# Vérifier le swap
swapon --show
```

---

## Dépannage

### Erreur "Out of memory"
```bash
# Vérifier le swap
swapon --show

# Si pas de swap, le créer:
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### L'API ne démarre pas
```bash
# Voir les erreurs
sudo journalctl -u cv-classifier -n 50

# Tester manuellement
cd ~/cv-classifier
source venv/bin/activate
python api/main.py
```

### Port 8000 non accessible
- Vérifier le Security Group dans AWS Console
- Ajouter une règle: Custom TCP, Port 8000, Source 0.0.0.0/0

---

## Coûts estimés (Free Tier)

| Ressource | Free Tier | Utilisation |
|-----------|-----------|-------------|
| EC2 t2.micro | 750h/mois | 24/7 = 720h ✅ |
| EBS 20GB | 30GB inclus | ✅ |
| Data transfer | 15GB/mois | Selon usage |

**Coût mensuel: $0** (pendant 12 mois)

---

## Mise à jour de l'application

```bash
cd ~/cv-classifier
git pull origin main
sudo systemctl restart cv-classifier
```
