"""
src/database/db_manager.py
Gestionnaire de base de données pour l'historique des classifications de CV
"""

import sqlite3
from datetime import datetime
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple
import pandas as pd


class CVDatabaseManager:
    """
    Gestionnaire de base de données SQLite pour stocker l'historique 
    des classifications de CV
    """
    
    def __init__(self, db_path: str = "data/cv_history.db"):
        """
        Initialiser le gestionnaire de base de données
        
        Args:
            db_path: Chemin vers le fichier de base de données SQLite
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Établir la connexion à la base de données"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Permet l'accès par nom de colonne
        self.cursor = self.conn.cursor()
    
    def _create_tables(self):
        """Créer les tables nécessaires si elles n'existent pas"""
        
        # Table principale des classifications
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cv_filename TEXT,
                cv_text TEXT,
                cv_text_preview TEXT,  -- Premiers 500 caractères
                predicted_category TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                all_probabilities TEXT,  -- JSON des probabilités pour toutes les catégories
                classification_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_used TEXT,
                model_version TEXT,
                processing_time_ms INTEGER,
                user_feedback TEXT,  -- Correct, Incorrect, NULL
                correct_category TEXT,  -- Si l'utilisateur corrige
                notes TEXT
            )
        """)
        
        # Table des compétences extraites
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS extracted_skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                classification_id INTEGER,
                skill_name TEXT NOT NULL,
                skill_category TEXT,  -- Technical, Soft, Language, etc.
                confidence REAL,
                FOREIGN KEY (classification_id) REFERENCES classifications(id)
            )
        """)
        
        # Table des statistiques quotidiennes
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date DATE PRIMARY KEY,
                total_classifications INTEGER DEFAULT 0,
                unique_categories INTEGER DEFAULT 0,
                avg_confidence REAL,
                most_common_category TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Index pour améliorer les performances
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_classification_date 
            ON classifications(classification_date)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predicted_category 
            ON classifications(predicted_category)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_skill_name 
            ON extracted_skills(skill_name)
        """)
        
        self.conn.commit()
        print(" Base de données initialisée avec succès")
    
    def add_classification(
        self,
        cv_text: str,
        predicted_category: str,
        confidence_score: float,
        cv_filename: Optional[str] = None,
        all_probabilities: Optional[Dict[str, float]] = None,
        model_used: Optional[str] = None,
        model_version: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        extracted_skills: Optional[List[Dict]] = None
    ) -> int:
        """
        Ajouter une nouvelle classification à la base de données
        
        Args:
            cv_text: Texte complet du CV
            predicted_category: Catégorie prédite
            confidence_score: Score de confiance (0-1)
            cv_filename: Nom du fichier CV (optionnel)
            all_probabilities: Probabilités pour toutes les catégories
            model_used: Nom du modèle utilisé
            model_version: Version du modèle
            processing_time_ms: Temps de traitement en millisecondes
            extracted_skills: Liste des compétences extraites
        
        Returns:
            ID de la classification créée
        """
        # Créer un aperçu du texte (premiers 500 caractères)
        cv_text_preview = cv_text[:500] if cv_text else None
        
        # Convertir les probabilités en JSON
        all_probs_json = json.dumps(all_probabilities) if all_probabilities else None
        
        # Insérer la classification
        self.cursor.execute("""
            INSERT INTO classifications (
                cv_filename, cv_text, cv_text_preview,
                predicted_category, confidence_score, all_probabilities,
                model_used, model_version, processing_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cv_filename, cv_text, cv_text_preview,
            predicted_category, confidence_score, all_probs_json,
            model_used, model_version, processing_time_ms
        ))
        
        classification_id = self.cursor.lastrowid
        
        # Ajouter les compétences extraites si fournies
        if extracted_skills:
            for skill in extracted_skills:
                self.add_skill(
                    classification_id,
                    skill.get('name'),
                    skill.get('category'),
                    skill.get('confidence')
                )
        
        self.conn.commit()
        
        # Mettre à jour les statistiques quotidiennes
        self._update_daily_stats()
        
        return classification_id
    
    def add_skill(
        self,
        classification_id: int,
        skill_name: str,
        skill_category: Optional[str] = None,
        confidence: Optional[float] = None
    ):
        """Ajouter une compétence extraite"""
        self.cursor.execute("""
            INSERT INTO extracted_skills (
                classification_id, skill_name, skill_category, confidence
            ) VALUES (?, ?, ?, ?)
        """, (classification_id, skill_name, skill_category, confidence))
        
        self.conn.commit()
    
    def update_feedback(
        self,
        classification_id: int,
        user_feedback: str,
        correct_category: Optional[str] = None,
        notes: Optional[str] = None
    ):
        """
        Mettre à jour le feedback utilisateur pour une classification
        
        Args:
            classification_id: ID de la classification
            user_feedback: "Correct", "Incorrect", ou autre
            correct_category: Catégorie correcte si feedback est "Incorrect"
            notes: Notes supplémentaires
        """
        self.cursor.execute("""
            UPDATE classifications
            SET user_feedback = ?,
                correct_category = ?,
                notes = ?
            WHERE id = ?
        """, (user_feedback, correct_category, notes, classification_id))
        
        self.conn.commit()
    
    def get_classification(self, classification_id: int) -> Optional[Dict]:
        """Récupérer une classification par ID"""
        self.cursor.execute("""
            SELECT * FROM classifications WHERE id = ?
        """, (classification_id,))
        
        row = self.cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_recent_classifications(self, limit: int = 10) -> List[Dict]:
        """Récupérer les classifications les plus récentes"""
        self.cursor.execute("""
            SELECT 
                id, cv_filename, cv_text_preview,
                predicted_category, confidence_score,
                classification_date, model_used
            FROM classifications
            ORDER BY classification_date DESC
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_classifications_by_category(self, category: str) -> List[Dict]:
        """Récupérer toutes les classifications d'une catégorie"""
        self.cursor.execute("""
            SELECT * FROM classifications
            WHERE predicted_category = ?
            ORDER BY classification_date DESC
        """, (category,))
        
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_classifications_by_date_range(
        self,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """
        Récupérer les classifications dans une période
        
        Args:
            start_date: Date de début (format: YYYY-MM-DD)
            end_date: Date de fin (format: YYYY-MM-DD)
        """
        self.cursor.execute("""
            SELECT * FROM classifications
            WHERE DATE(classification_date) BETWEEN ? AND ?
            ORDER BY classification_date DESC
        """, (start_date, end_date))
        
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_statistics(self) -> Dict:
        """Obtenir des statistiques globales"""
        stats = {}
        
        # Total de classifications
        self.cursor.execute("SELECT COUNT(*) FROM classifications")
        stats['total_classifications'] = self.cursor.fetchone()[0]
        
        # Confiance moyenne
        self.cursor.execute("SELECT AVG(confidence_score) FROM classifications")
        stats['avg_confidence'] = round(self.cursor.fetchone()[0] or 0, 4)
        
        # Distribution par catégorie
        self.cursor.execute("""
            SELECT predicted_category, COUNT(*) as count
            FROM classifications
            GROUP BY predicted_category
            ORDER BY count DESC
        """)
        stats['category_distribution'] = [
            {'category': row[0], 'count': row[1]}
            for row in self.cursor.fetchall()
        ]
        
        # Précision basée sur le feedback
        self.cursor.execute("""
            SELECT 
                COUNT(CASE WHEN user_feedback = 'Correct' THEN 1 END) as correct,
                COUNT(CASE WHEN user_feedback = 'Incorrect' THEN 1 END) as incorrect,
                COUNT(CASE WHEN user_feedback IS NOT NULL THEN 1 END) as total_feedback
            FROM classifications
        """)
        row = self.cursor.fetchone()
        if row[2] > 0:
            stats['accuracy_from_feedback'] = round(row[0] / row[2], 4)
        else:
            stats['accuracy_from_feedback'] = None
        
        # Top 10 compétences
        self.cursor.execute("""
            SELECT skill_name, COUNT(*) as count
            FROM extracted_skills
            GROUP BY skill_name
            ORDER BY count DESC
            LIMIT 10
        """)
        stats['top_skills'] = [
            {'skill': row[0], 'count': row[1]}
            for row in self.cursor.fetchall()
        ]
        
        return stats
    
    def get_category_stats(self, category: str) -> Dict:
        """Obtenir des statistiques pour une catégorie spécifique"""
        stats = {'category': category}
        
        # Nombre total
        self.cursor.execute("""
            SELECT COUNT(*) FROM classifications
            WHERE predicted_category = ?
        """, (category,))
        stats['total_count'] = self.cursor.fetchone()[0]
        
        # Confiance moyenne
        self.cursor.execute("""
            SELECT AVG(confidence_score)
            FROM classifications
            WHERE predicted_category = ?
        """, (category,))
        stats['avg_confidence'] = round(self.cursor.fetchone()[0] or 0, 4)
        
        # Compétences les plus fréquentes pour cette catégorie
        self.cursor.execute("""
            SELECT es.skill_name, COUNT(*) as count
            FROM extracted_skills es
            JOIN classifications c ON es.classification_id = c.id
            WHERE c.predicted_category = ?
            GROUP BY es.skill_name
            ORDER BY count DESC
            LIMIT 10
        """, (category,))
        stats['top_skills'] = [
            {'skill': row[0], 'count': row[1]}
            for row in self.cursor.fetchall()
        ]
        
        return stats
    
    def _update_daily_stats(self):
        """Mettre à jour les statistiques quotidiennes"""
        today = datetime.now().date()
        
        # Calculer les stats du jour
        self.cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT predicted_category) as unique_cats,
                AVG(confidence_score) as avg_conf,
                predicted_category as most_common
            FROM classifications
            WHERE DATE(classification_date) = ?
            GROUP BY DATE(classification_date)
            ORDER BY COUNT(*) DESC
            LIMIT 1
        """, (today,))
        
        row = self.cursor.fetchone()
        if row:
            self.cursor.execute("""
                INSERT OR REPLACE INTO daily_stats (
                    date, total_classifications, unique_categories,
                    avg_confidence, most_common_category
                ) VALUES (?, ?, ?, ?, ?)
            """, (today, row[0], row[1], round(row[2], 4), row[3]))
            
            self.conn.commit()
    
    def export_to_csv(self, output_path: str, include_skills: bool = False):
        """
        Exporter les données vers un fichier CSV
        
        Args:
            output_path: Chemin du fichier CSV de sortie
            include_skills: Inclure les compétences extraites
        """
        # Exporter les classifications
        df = pd.read_sql_query("""
            SELECT 
                id, cv_filename, predicted_category, confidence_score,
                classification_date, model_used, user_feedback, correct_category
            FROM classifications
            ORDER BY classification_date DESC
        """, self.conn)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f" Classifications exportées vers: {output_path}")
        
        # Exporter les compétences si demandé
        if include_skills:
            skills_path = output_path.replace('.csv', '_skills.csv')
            df_skills = pd.read_sql_query("""
                SELECT 
                    es.classification_id, es.skill_name, es.skill_category,
                    c.predicted_category
                FROM extracted_skills es
                JOIN classifications c ON es.classification_id = c.id
                ORDER BY es.classification_id DESC
            """, self.conn)
            
            df_skills.to_csv(skills_path, index=False, encoding='utf-8')
            print(f" Compétences exportées vers: {skills_path}")
    
    def export_statistics_report(self, output_path: str):
        """Exporter un rapport de statistiques complet"""
        stats = self.get_statistics()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAPPORT DE STATISTIQUES - CLASSIFICATION DE CV\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Date du rapport: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f" STATISTIQUES GLOBALES\n")
            f.write(f"{'─' * 80}\n")
            f.write(f"Total de classifications: {stats['total_classifications']}\n")
            f.write(f"Confiance moyenne: {stats['avg_confidence']:.2%}\n")
            if stats['accuracy_from_feedback']:
                f.write(f"Précision (basée sur feedback): {stats['accuracy_from_feedback']:.2%}\n")
            f.write("\n")
            
            f.write(f" DISTRIBUTION PAR CATÉGORIE\n")
            f.write(f"{'─' * 80}\n")
            for item in stats['category_distribution'][:10]:
                f.write(f"{item['category']:30s} {item['count']:>5d} classifications\n")
            f.write("\n")
            
            f.write(f"🔧 TOP 10 COMPÉTENCES\n")
            f.write(f"{'─' * 80}\n")
            for item in stats['top_skills']:
                f.write(f"{item['skill']:30s} {item['count']:>5d} occurrences\n")
            f.write("\n")
        
        print(f" Rapport exporté vers: {output_path}")
    
    def search_by_skill(self, skill_name: str) -> List[Dict]:
        """Rechercher toutes les classifications contenant une compétence"""
        self.cursor.execute("""
            SELECT DISTINCT c.*
            FROM classifications c
            JOIN extracted_skills es ON c.id = es.classification_id
            WHERE es.skill_name LIKE ?
            ORDER BY c.classification_date DESC
        """, (f"%{skill_name}%",))
        
        return [dict(row) for row in self.cursor.fetchall()]
    
    def delete_classification(self, classification_id: int):
        """Supprimer une classification et ses compétences associées"""
        # Supprimer les compétences
        self.cursor.execute("""
            DELETE FROM extracted_skills
            WHERE classification_id = ?
        """, (classification_id,))
        
        # Supprimer la classification
        self.cursor.execute("""
            DELETE FROM classifications
            WHERE id = ?
        """, (classification_id,))
        
        self.conn.commit()
        print(f" Classification {classification_id} supprimée")
    
    def close(self):
        """Fermer la connexion à la base de données"""
        if self.conn:
            self.conn.close()
            print(" Connexion à la base de données fermée")
    
    def __enter__(self):
        """Support pour le context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Fermeture automatique avec context manager"""
        self.close()


# ============================================
# EXEMPLE D'UTILISATION
# ============================================

if __name__ == "__main__":
    # Créer le gestionnaire
    with CVDatabaseManager("data/cv_history.db") as db:
        
        # Exemple 1: Ajouter une classification
        classification_id = db.add_classification(
            cv_text="Experienced Python developer with 5 years in ML...",
            predicted_category="Data Scientist",
            confidence_score=0.95,
            cv_filename="john_doe_cv.pdf",
            all_probabilities={
                "Data Scientist": 0.95,
                "Software Engineer": 0.03,
                "ML Engineer": 0.02
            },
            model_used="Random_Forest",
            model_version="1.0",
            processing_time_ms=150,
            extracted_skills=[
                {"name": "Python", "category": "Programming", "confidence": 0.98},
                {"name": "Machine Learning", "category": "Technical", "confidence": 0.95},
                {"name": "TensorFlow", "category": "Framework", "confidence": 0.90}
            ]
        )
        
        print(f"\n Classification ajoutée avec ID: {classification_id}")
        
        # Exemple 2: Obtenir des statistiques
        stats = db.get_statistics()
        print(f"\n Statistiques:")
        print(f"   Total: {stats['total_classifications']}")
        print(f"   Confiance moyenne: {stats['avg_confidence']:.2%}")
        
        # Exemple 3: Exporter un rapport
        db.export_statistics_report("outputs/reports/db_statistics.txt")
