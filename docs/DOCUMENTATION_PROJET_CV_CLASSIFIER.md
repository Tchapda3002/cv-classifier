# CV Classifier - Documentation Technique Complete

## Guide Complet du Systeme de Classification Automatique de CV

**Version**: 2.2.0
**Date**: Fevrier 2026
**Auteur**: Projet NLP Final

---

# Table des Matieres

1. [Introduction et Vue d'Ensemble](#1-introduction-et-vue-densemble)
2. [Architecture du Systeme](#2-architecture-du-systeme)
3. [Preprocessing du Texte - Comment le CV est Nettoye](#3-preprocessing-du-texte)
4. [Extraction de Caracteristiques - TF-IDF](#4-extraction-de-caracteristiques-tf-idf)
5. [Classification - Le Modele de Machine Learning](#5-classification-le-modele-de-machine-learning)
6. [Calcul de la Confiance](#6-calcul-de-la-confiance)
7. [Extraction des Competences](#7-extraction-des-competences)
8. [Le Chatbot IA](#8-le-chatbot-ia)
9. [Limites et Considerations](#9-limites-et-considerations)
10. [Glossaire](#10-glossaire)

---

# 1. Introduction et Vue d'Ensemble

## Qu'est-ce que CV Classifier ?

CV Classifier est un systeme intelligent qui analyse automatiquement le contenu d'un CV (Curriculum Vitae) et determine la categorie professionnelle la plus appropriee pour le candidat. Par exemple, si vous uploadez un CV contenant des competences en Python, Machine Learning et Data Analysis, le systeme le classifiera probablement dans la categorie "Data Scientist".

## Comment ca fonctionne en resume ?

```
CV (texte brut) --> Nettoyage --> Vectorisation --> Classification --> Categorie + Confiance
```

**Etapes simplifiees** :
1. Le systeme recoit le texte du CV
2. Il nettoie le texte (supprime emails, URLs, ponctuations, etc.)
3. Il transforme le texte en nombres (vecteurs)
4. Le modele ML analyse ces nombres
5. Il renvoie une categorie avec un pourcentage de confiance

---

# 2. Architecture du Systeme

## Schema Global

```
+------------------+     +-------------------+     +------------------+
|   FRONTEND       |     |      API          |     |    MODELES ML    |
|   (HTML/JS)      | --> |   (FastAPI)       | --> |  (Scikit-learn)  |
+------------------+     +-------------------+     +------------------+
                               |
                    +----------+----------+
                    |          |          |
              +-----v----+ +---v---+ +----v-----+
              | Text     | | Skills| | Chatbot  |
              | Cleaner  | | Detect| | (HF API) |
              +----------+ +-------+ +----------+
```

## Composants Principaux

| Composant | Role | Fichier |
|-----------|------|---------|
| API REST | Point d'entree des requetes | `api/main.py` |
| Text Cleaner | Nettoyage du texte | `src/preprocessing/text_cleaner.py` |
| Pipeline ML | Classification | `models/cv_classifier_pipeline.pkl` |
| Skills Detector | Detection competences | `src/skills_extraction/skills_detector.py` |
| Chatbot | Questions en langage naturel | `src/chatbot/cv_chatbot.py` |

---

# 3. Preprocessing du Texte

## Pourquoi nettoyer le texte ?

Un CV contient beaucoup d'informations "bruitees" qui n'aident pas le modele a comprendre le profil du candidat :
- Emails (john.doe@email.com)
- Numeros de telephone (+33 6 12 34 56 78)
- URLs (https://linkedin.com/in/johndoe)
- Dates et chiffres non pertinents

## Les 12 Etapes de Nettoyage

Le systeme applique ces transformations dans l'ordre :

### Etape 1 : Conversion en Minuscules
```
AVANT : "Senior SOFTWARE Engineer"
APRES : "senior software engineer"
```
**Pourquoi ?** Pour que "Python" et "python" soient reconnus comme le meme mot.

### Etape 2 : Suppression des URLs
```
AVANT : "Mon portfolio: https://monsite.com/portfolio"
APRES : "Mon portfolio: "
```
**Pourquoi ?** Les URLs n'apportent pas d'information sur les competences.

### Etape 3 : Suppression des Emails
```
AVANT : "Contactez-moi: jean.dupont@gmail.com"
APRES : "Contactez-moi: "
```

### Etape 4 : Suppression des Numeros de Telephone
```
AVANT : "Tel: +33 6 12 34 56 78"
APRES : "Tel: "
```

### Etape 5 : Suppression des Nombres (optionnel)
```
AVANT : "5 ans d'experience en 2023"
APRES : "ans d'experience en"
```
**Note** : Cette option est configurable car parfois les annees d'experience sont importantes.

### Etape 6 : Suppression de la Ponctuation
```
AVANT : "Python, Java, C++!"
APRES : "Python Java C"
```

### Etape 7 : Normalisation des Espaces
```
AVANT : "Python    Java     C"
APRES : "Python Java C"
```

### Etape 8 : Tokenisation
Le texte est decoupe en mots individuels (tokens) :
```
AVANT : "Python Java C"
APRES : ["Python", "Java", "C"]
```

### Etape 9 : Suppression des Stopwords
Les mots communs sans valeur semantique sont retires :
```
AVANT : ["the", "developer", "is", "working", "on", "python"]
APRES : ["developer", "working", "python"]
```
**Stopwords supprimes** : "the", "is", "on", "a", "an", "and", "or", etc.

### Etape 10 : Suppression des Tokens Courts
Les mots de moins de 3 caracteres sont retires :
```
AVANT : ["a", "developer", "in", "it"]
APRES : ["developer"]
```

### Etape 11 : Lemmatisation
Les mots sont reduits a leur forme de base :
```
AVANT : ["developing", "developed", "developer"]
APRES : ["develop", "develop", "developer"]
```
**Explication** : "working" devient "work", "machines" devient "machine".

### Etape 12 : Reconstruction
Les tokens sont rassembles en une chaine :
```
AVANT : ["python", "machine", "learning", "developer"]
APRES : "python machine learning developer"
```

## Exemple Complet de Nettoyage

**TEXTE ORIGINAL** :
```
John Doe
Email: john.doe@example.com | Phone: +1-555-123-4567
Website: https://johndoe.com

PROFESSIONAL SUMMARY
Experienced Software Developer with 5+ years in Python, JavaScript,
and cloud technologies. Strong background in machine learning!!!

SKILLS: Python, JavaScript, React, Docker, AWS, SQL, MongoDB
```

**TEXTE NETTOYE** :
```
professional summary experienced software developer year python
javascript cloud technology strong background machine learning
skill python javascript react docker aws sql mongodb
```

**Reduction** : De 350 caracteres a 180 caracteres (reduction de 48%)

---

# 4. Extraction de Caracteristiques - TF-IDF

## Le Probleme : Les Ordinateurs ne Comprennent pas les Mots

Un modele de machine learning ne peut pas traiter directement du texte. Il ne comprend que les nombres. Nous devons donc transformer le texte en representation numerique.

## Qu'est-ce que TF-IDF ?

**TF-IDF** = Term Frequency - Inverse Document Frequency

C'est une methode mathematique pour mesurer l'importance de chaque mot dans un document par rapport a une collection de documents.

### TF (Term Frequency) - Frequence du Terme

**Formule simplifiee** :
```
TF = Nombre de fois que le mot apparait dans le CV / Nombre total de mots dans le CV
```

**Exemple** :
- Si "python" apparait 5 fois dans un CV de 100 mots
- TF("python") = 5/100 = 0.05

### IDF (Inverse Document Frequency)

**Formule simplifiee** :
```
IDF = log(Nombre total de CVs / Nombre de CVs contenant ce mot)
```

**Exemple** :
- Sur 1000 CVs, 100 contiennent le mot "python"
- IDF("python") = log(1000/100) = log(10) = 2.3

**Intuition** :
- Si un mot apparait dans TOUS les CVs (comme "experience"), son IDF sera faible (peu distinctif)
- Si un mot apparait dans PEU de CVs (comme "kubernetes"), son IDF sera eleve (tres distinctif)

### TF-IDF Final

```
TF-IDF = TF x IDF
```

**Exemple pour "python"** :
- TF = 0.05, IDF = 2.3
- TF-IDF = 0.05 x 2.3 = 0.115

## Configuration Utilisee dans le Projet

```python
TfidfVectorizer(
    max_features=5000,    # Garde les 5000 mots les plus importants
    ngram_range=(1, 2),   # Considere les mots seuls ET les paires
    min_df=2,             # Ignore les mots qui apparaissent dans moins de 2 CVs
    max_df=0.95           # Ignore les mots qui apparaissent dans plus de 95% des CVs
)
```

### Explication des N-grams

Avec `ngram_range=(1, 2)`, le systeme considere :
- **Unigrams** (1 mot) : "machine", "learning"
- **Bigrams** (2 mots) : "machine learning"

**Pourquoi ?** Car "machine learning" a plus de sens que "machine" et "learning" separement.

## Resultat : Le Vecteur

Apres TF-IDF, chaque CV devient un vecteur de 5000 dimensions :
```
CV1 = [0.0, 0.12, 0.0, 0.08, 0.0, ..., 0.15]  # 5000 valeurs
CV2 = [0.05, 0.0, 0.11, 0.0, 0.09, ..., 0.0]  # 5000 valeurs
```

Chaque position correspond a un mot, et la valeur represente son importance TF-IDF.

---

# 5. Classification - Le Modele de Machine Learning

## Quel Modele est Utilise ?

Le systeme utilise un **Random Forest Classifier** (Foret Aleatoire).

## Comment Fonctionne Random Forest ?

### Analogie Simple : Le Conseil des Experts

Imaginez que vous demandez a 100 experts de voter pour determiner la profession d'un candidat :

1. Chaque expert (arbre) regarde le CV differemment
2. Chaque expert donne son vote (sa prediction)
3. La categorie avec le plus de votes gagne

### Fonctionnement Technique

1. **Creation de multiples arbres de decision**
   - 100 arbres sont crees (parametre n_estimators)
   - Chaque arbre est entraine sur un sous-ensemble aleatoire des donnees

2. **Echantillonnage Bootstrap**
   - Chaque arbre utilise environ 63% des donnees (tirage avec remise)
   - Les 37% restants servent a evaluer l'arbre

3. **Selection aleatoire des caracteristiques**
   - A chaque noeud, seul un sous-ensemble de mots est considere
   - Cela evite que tous les arbres se ressemblent

4. **Vote majoritaire**
   - Chaque arbre vote pour une categorie
   - La categorie majoritaire est selectionnee

### Exemple de Decision d'un Arbre

```
                     [CV du candidat]
                           |
                 contient "python" ?
                    /            \
                  OUI            NON
                   |              |
          contient "tensorflow" ?  contient "java" ?
              /        \              /        \
            OUI        NON          OUI        NON
             |          |            |          |
        Data Scientist  ...    Software Eng    ...
```

## Les 25 Categories

Le modele peut classifier les CVs dans ces categories :

| # | Categorie |
|---|-----------|
| 1 | Advocate |
| 2 | Arts |
| 3 | Automation Testing |
| 4 | Blockchain |
| 5 | Business Analyst |
| 6 | Civil Engineer |
| 7 | Data Science |
| 8 | Database |
| 9 | DevOps Engineer |
| 10 | DotNet Developer |
| 11 | ETL Developer |
| 12 | Electrical Engineering |
| 13 | HR |
| 14 | Hadoop |
| 15 | Health and Fitness |
| 16 | Java Developer |
| 17 | Mechanical Engineer |
| 18 | Network Security Engineer |
| 19 | Operations Manager |
| 20 | PMO |
| 21 | Python Developer |
| 22 | SAP Developer |
| 23 | Sales |
| 24 | Testing |
| 25 | Web Designing |

---

# 6. Calcul de la Confiance

## Comment le Systeme Calcule-t-il la Confiance ?

La confiance n'est pas une estimation subjective mais un **calcul mathematique** base sur les votes des arbres.

### Methode : predict_proba()

```python
probabilities = pipeline.predict_proba([cv_text])[0]
confidence = probabilities.max()
```

### Explication Detaillee

1. **Chaque arbre vote pour une categorie**
   - Arbre 1 vote "Data Scientist"
   - Arbre 2 vote "Data Scientist"
   - Arbre 3 vote "Python Developer"
   - ... (100 arbres au total)

2. **Calcul des probabilites**
   ```
   P("Data Scientist") = 75 arbres / 100 arbres = 0.75 (75%)
   P("Python Developer") = 20 arbres / 100 arbres = 0.20 (20%)
   P("Software Engineer") = 5 arbres / 100 arbres = 0.05 (5%)
   ```

3. **La confiance = probabilite maximale**
   ```
   Confiance = max(0.75, 0.20, 0.05) = 0.75 = 75%
   ```

### Interpretation des Niveaux de Confiance

| Confiance | Interpretation | Recommandation |
|-----------|----------------|----------------|
| > 90% | Tres haute | Fiable |
| 70-90% | Haute | Probablement correct |
| 50-70% | Moyenne | Verification recommandee |
| < 50% | Basse | CV ambigu, plusieurs profils possibles |

### Exemple de Reponse Complete

```json
{
    "category": "Data Scientist",
    "confidence": 0.87,
    "all_probabilities": {
        "Data Scientist": 0.87,
        "Python Developer": 0.08,
        "Machine Learning Engineer": 0.03,
        "Software Engineer": 0.02,
        "Other categories": 0.00
    }
}
```

---

# 7. Extraction des Competences

## Comment les Competences sont-elles Detectees ?

Le systeme utilise une **detection par correspondance de motifs** (Pattern Matching) avec une base de donnees de competences.

## La Base de Competences

Le systeme contient plusieurs categories de competences :

### 1. Competences Techniques (par domaine)

```python
technical_skills = {
    'Machine Learning': [
        'machine learning', 'deep learning', 'neural networks',
        'nlp', 'computer vision', 'tensorflow', 'pytorch'
    ],
    'Data Science': [
        'data science', 'data analysis', 'statistics',
        'data visualization', 'regression', 'clustering'
    ],
    'Programming': [
        'python', 'java', 'javascript', 'c++', 'sql'
    ],
    # ... autres categories
}
```

### 2. Soft Skills

```python
soft_skills = [
    'communication', 'teamwork', 'leadership',
    'problem solving', 'critical thinking', 'creativity'
]
```

### 3. Frameworks et Outils

```python
frameworks = {
    'ML/DL Frameworks': ['tensorflow', 'pytorch', 'scikit-learn', 'keras'],
    'Web Frameworks': ['react', 'angular', 'vue', 'django', 'flask'],
    # ...
}
```

## Algorithme de Detection

### Etape 1 : Recherche de Motifs (Pattern Matching)

```python
def _find_skill(self, skill: str, text: str) -> bool:
    # Utilise les expressions regulieres avec word boundaries
    pattern = r'\b' + re.escape(skill) + r'\b'
    return bool(re.search(pattern, text, re.IGNORECASE))
```

**Word Boundaries** (`\b`) : Evite les faux positifs
- "java" ne matchera PAS "javascript"
- "python" ne matchera PAS "pythonic"

### Etape 2 : Calcul du Score de Confiance

La confiance de chaque competence est calculee sur deux criteres :

#### A. Score de Frequence (70% du poids)

```python
# Compter les occurrences
count = len(re.findall(pattern, text))

# Score plafonne a 1.0
frequency_score = min(count * 0.2, 1.0)
```

**Exemple** :
- "python" apparait 3 fois --> score = min(3 * 0.2, 1.0) = 0.6
- "python" apparait 5 fois --> score = min(5 * 0.2, 1.0) = 1.0

#### B. Score de Contexte (30% du poids)

Le systeme verifie si la competence apparait pres de mots-cles contextuels :

```python
context_keywords = ['skills', 'expertise', 'proficient',
                    'experience', 'knowledge', 'technologies']
```

**Exemple** :
- Si "python" apparait dans "Strong Python skills" --> bonus +0.1
- Si "python" apparait dans "Technologies: Python, Java" --> bonus +0.1

#### C. Score Final

```python
final_score = (frequency_score * 0.7) + (context_score * 0.3)
```

### Exemple de Resultats

```json
{
    "technical_skills": [
        {"skill": "python", "category": "Programming", "confidence": 0.85},
        {"skill": "machine learning", "category": "Machine Learning", "confidence": 0.72},
        {"skill": "sql", "category": "Databases", "confidence": 0.45}
    ],
    "frameworks": [
        {"framework": "tensorflow", "category": "ML/DL Frameworks", "confidence": 0.68}
    ],
    "soft_skills": [
        {"skill": "leadership", "confidence": 0.52}
    ]
}
```

---

# 8. Le Chatbot IA

## Fonctionnement General

Le chatbot permet de poser des questions en langage naturel sur un CV analyse.

## Architecture

```
Question Utilisateur --> API Chatbot --> HuggingFace API --> Reponse
                              |
                         CV en contexte
```

## Le Modele Utilise

Le systeme utilise **Llama 3.2 3B Instruct** via l'API HuggingFace Inference :

| Propriete | Valeur |
|-----------|--------|
| Modele | meta-llama/Llama-3.2-3B-Instruct |
| Parametres | 3 milliards |
| Type | Instruction-tuned (optimise pour suivre des instructions) |
| API | HuggingFace Inference (gratuit) |

## Comment ca Marche ?

### 1. Construction du Prompt

Le systeme construit un message structure :

```python
messages = [
    {
        "role": "system",
        "content": """Tu es un assistant specialise dans l'analyse de CV.
                     Tu dois repondre UNIQUEMENT sur le CV fourni.
                     Sois concis et precis. Reponds en francais."""
    },
    {
        "role": "user",
        "content": f"Voici le CV a analyser:\n\n{cv_text[:3000]}\n\n---"
    },
    {
        "role": "user",
        "content": question  # La question de l'utilisateur
    }
]
```

### 2. Appel a l'API

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=messages,
    max_tokens=500,
    temperature=0.7
)
```

### 3. Gestion de l'Historique

Le chatbot garde en memoire les 3 derniers echanges pour maintenir le contexte :

```python
for exchange in self.conversation_history[-3:]:
    messages.append({"role": "user", "content": exchange['question']})
    messages.append({"role": "assistant", "content": exchange['answer']})
```

## Fallback : SimpleCVChatbot

Si l'API HuggingFace n'est pas disponible, un chatbot simple prend le relais :

```python
class SimpleCVChatbot:
    """Chatbot base sur des regles simples (sans IA)"""

    def ask(self, question):
        if "competence" in question.lower():
            return self._extract_skills()
        elif "experience" in question.lower():
            return self._extract_experience()
        # ...
```

Ce fallback utilise des patterns predefinies au lieu de l'IA.

---

# 9. Limites et Considerations

## 9.1 Limites du Modele de Classification

### A. Dependance aux Donnees d'Entrainement

**Probleme** : Le modele a ete entraine sur 962 CVs. Ses predictions sont limitees a ce qu'il a "vu".

**Consequences** :
- Metiers emergents (ex: "Prompt Engineer") non reconnus
- Profils atypiques mal classes
- Biais vers les profils surrepresentes dans les donnees

### B. Sensibilite au Vocabulaire

**Probleme** : Le modele TF-IDF ne comprend pas les synonymes.

**Exemple** :
- "ML Engineer" et "Machine Learning Engineer" sont traites differemment
- "Dev" vs "Developer" peuvent donner des resultats differents

### C. Pas de Comprehension Semantique

**Probleme** : Le modele ne "comprend" pas vraiment le sens du texte.

**Exemple** :
- "Je ne connais PAS Python" peut etre interprete comme une competence Python
- Le contexte negatif n'est pas pris en compte

## 9.2 Limites de l'Extraction de Competences

### A. Detection par Mots-Cles

**Probleme** : Le systeme cherche des correspondances exactes.

**Consequences** :
- "TensorFlow" trouve, mais "TF" non reconnu
- Fautes d'orthographe non gerees ("Pytohn" au lieu de "Python")
- Abreviations non standard ignorees

### B. Faux Positifs Possibles

**Exemple** :
- "I worked with Java developers" --> "Java" detecte mais le candidat n'est pas developpeur Java
- "Managed a team using Agile" --> "Agile" detecte mais niveau reel inconnu

### C. Base de Competences Statique

La liste des competences est predefinie et peut devenir obsolete face aux nouvelles technologies.

## 9.3 Limites du Chatbot

### A. Dependance a l'API Externe

**Probleme** : Le chatbot necessite une connexion a HuggingFace.

**Consequences** :
- Latence reseau (1-3 secondes par reponse)
- Indisponibilite possible de l'API
- Limites de rate (nombre de requetes)

### B. Hallucinations Possibles

**Probleme** : Le modele LLM peut "inventer" des informations.

**Exemple** :
- Question : "Quel est le salaire du candidat ?"
- Reponse possible : "Le candidat gagne environ 50000 euros" (invente)

### C. Contexte Limite

Le CV est tronque a 3000 caracteres pour respecter les limites de tokens du modele.

## 9.4 Consideratios de Securite et Confidentialite

### A. Donnees Sensibles

Les CVs contiennent des informations personnelles :
- Noms, adresses, numeros de telephone
- Historique professionnel
- Parfois photos et age

**Recommandation** : Ne pas stocker les CVs plus longtemps que necessaire.

### B. Utilisation des APIs Externes

L'utilisation de HuggingFace API signifie que le texte du CV transite par leurs serveurs.

**Recommandation** : Pour des usages sensibles, envisager des modeles locaux.

## 9.5 Metriques de Performance

### Performances Actuelles (sur le jeu de test)

| Metrique | Valeur |
|----------|--------|
| Accuracy | 100% |
| F1-Score (macro) | 98.96% |
| Precision | 99.2% |
| Recall | 98.8% |

### Avertissement Important

Ces metriques tres elevees peuvent indiquer :
1. Un jeu de test tres similaire aux donnees d'entrainement
2. Un possible sur-apprentissage (overfitting)
3. Une fuite de donnees (data leakage) corrigee dans la version actuelle

**Les performances reelles sur des CVs "dans la nature" seront probablement inferieures.**

---

# 10. Glossaire

| Terme | Definition Simple |
|-------|-------------------|
| **API** | Interface permettant a des programmes de communiquer entre eux |
| **Classification** | Tache consistant a attribuer une categorie a un element |
| **Confiance** | Probabilite que la prediction soit correcte (entre 0 et 1) |
| **Data Leakage** | Fuite de donnees : quand des infos du test sont utilisees pendant l'entrainement |
| **F1-Score** | Mesure combinant precision et rappel (moyenne harmonique) |
| **FastAPI** | Framework Python pour creer des APIs web |
| **HuggingFace** | Plateforme de modeles de Machine Learning |
| **Lemmatisation** | Reduire un mot a sa forme de base ("working" --> "work") |
| **LLM** | Large Language Model - Grand modele de langage (GPT, Llama, etc.) |
| **Machine Learning** | Apprentissage automatique par les donnees |
| **N-gram** | Sequence de N mots consecutifs |
| **NLP** | Natural Language Processing - Traitement du langage naturel |
| **Overfitting** | Sur-apprentissage : le modele memorise au lieu de generaliser |
| **Pipeline** | Enchainement automatique d'etapes de traitement |
| **Precision** | % des predictions positives qui sont correctes |
| **Preprocessing** | Pre-traitement : nettoyage des donnees avant analyse |
| **Random Forest** | Algorithme ML utilisant plusieurs arbres de decision |
| **Recall** | % des vrais positifs qui sont detectes |
| **Regex** | Expression reguliere : motif de recherche de texte |
| **REST** | Style d'architecture pour les APIs web |
| **Sklearn** | Scikit-learn : bibliotheque Python de Machine Learning |
| **Stopwords** | Mots vides (le, la, de, et) sans valeur semantique |
| **TF-IDF** | Methode de ponderation des mots par frequence et rarete |
| **Token** | Unite de texte (mot ou sous-mot) |
| **Vectorisation** | Transformation du texte en nombres |

---

# Conclusion

CV Classifier est un systeme complet qui combine :
- **Pre-traitement NLP** pour nettoyer les textes
- **TF-IDF** pour transformer le texte en vecteurs
- **Random Forest** pour classifier les CVs
- **Pattern Matching** pour extraire les competences
- **LLM via API** pour le chatbot conversationnel

Bien que les performances soient excellentes sur les donnees de test, il est important de garder en tete les limites inherentes a tout systeme de Machine Learning : dependance aux donnees d'entrainement, pas de comprehension semantique profonde, et necessite de mise a jour reguliere.

---

**Document genere le** : Fevrier 2026
**Version du systeme** : 2.2.0
**Contact** : [Votre email]
