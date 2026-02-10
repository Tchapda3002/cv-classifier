"""
Tests d'intégration pour l'API FastAPI
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def client():
    """Créer un client de test FastAPI"""
    from api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests pour l'endpoint /health"""

    def test_health_returns_200(self, client):
        """Test que /health retourne 200"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Test la structure de la réponse /health"""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "vectorizer_loaded" in data

    def test_health_models_loaded(self, client):
        """Test que les modèles sont chargés"""
        response = client.get("/health")
        data = response.json()

        assert data["model_loaded"] is True
        assert data["vectorizer_loaded"] is True
        assert data["label_encoder_loaded"] is True


class TestRootEndpoint:
    """Tests pour l'endpoint racine /"""

    def test_root_returns_200(self, client):
        """Test que / retourne 200"""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_version(self, client):
        """Test que la version est présente"""
        response = client.get("/")
        data = response.json()

        assert "version" in data


class TestPredictEndpoint:
    """Tests pour l'endpoint /predict"""

    def test_predict_valid_cv(self, client):
        """Test prédiction avec un CV valide"""
        cv_text = """
        Senior Python Developer with 5 years experience.
        Skills: Python, Django, FastAPI, PostgreSQL, Docker.
        Experience in machine learning and data science.
        """

        response = client.post(
            "/predict",
            json={"resume_text": cv_text}
        )

        assert response.status_code == 200
        data = response.json()
        assert "category" in data
        assert "confidence" in data
        assert data["confidence"] > 0

    def test_predict_empty_cv(self, client):
        """Test prédiction avec CV vide"""
        response = client.post(
            "/predict",
            json={"resume_text": ""}
        )

        assert response.status_code == 400

    def test_predict_short_cv(self, client):
        """Test prédiction avec CV trop court"""
        response = client.post(
            "/predict",
            json={"resume_text": "hello"}
        )

        assert response.status_code == 400

    def test_predict_java_developer(self, client):
        """Test prédiction pour Java Developer"""
        cv_text = """
        Java Developer with expertise in Spring Boot, Hibernate.
        Experience with microservices, REST APIs, Maven, Jenkins.
        Strong background in enterprise software development.
        """

        response = client.post(
            "/predict",
            json={"resume_text": cv_text}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "Java Developer"

    def test_predict_with_probabilities(self, client):
        """Test prédiction avec toutes les probabilités"""
        cv_text = "Python developer with machine learning experience"

        response = client.post(
            "/predict?include_all_probabilities=true",
            json={"resume_text": cv_text}
        )

        assert response.status_code == 200
        data = response.json()
        assert "all_probabilities" in data
        assert len(data["all_probabilities"]) == 25  # 25 catégories


class TestCategoriesEndpoint:
    """Tests pour l'endpoint /categories"""

    def test_categories_returns_200(self, client):
        """Test que /categories retourne 200"""
        response = client.get("/categories")
        assert response.status_code == 200

    def test_categories_count(self, client):
        """Test le nombre de catégories"""
        response = client.get("/categories")
        data = response.json()

        assert data["total_categories"] == 25
        assert len(data["categories"]) == 25

    def test_categories_contains_expected(self, client):
        """Test que les catégories attendues sont présentes"""
        response = client.get("/categories")
        data = response.json()

        expected = ["Java Developer", "Python Developer", "Data Science", "HR"]
        for cat in expected:
            assert cat in data["categories"]


class TestModelInfoEndpoint:
    """Tests pour l'endpoint /model-info"""

    def test_model_info_returns_200(self, client):
        """Test que /model-info retourne 200"""
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_structure(self, client):
        """Test la structure de /model-info"""
        response = client.get("/model-info")
        data = response.json()

        assert "model_type" in data
        assert "n_features" in data
        assert "n_categories" in data

    def test_model_is_random_forest(self, client):
        """Test que le modèle est Random Forest"""
        response = client.get("/model-info")
        data = response.json()

        assert data["model_type"] == "RandomForestClassifier"


class TestBatchPredictEndpoint:
    """Tests pour l'endpoint /batch-predict"""

    def test_batch_predict_multiple_cvs(self, client):
        """Test prédiction en batch"""
        cvs = [
            "Python developer with ML experience",
            "Java developer Spring Boot microservices",
            "HR manager recruitment talent acquisition"
        ]

        response = client.post(
            "/batch-predict",
            json={"resumes": cvs}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 3
        assert len(data["predictions"]) == 3

    def test_batch_predict_empty_list(self, client):
        """Test batch avec liste vide"""
        response = client.post(
            "/batch-predict",
            json={"resumes": []}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 0
