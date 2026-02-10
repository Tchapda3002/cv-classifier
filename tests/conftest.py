"""
Configuration globale pour pytest
"""

import pytest
import sys
from pathlib import Path

# Ajouter les chemins du projet
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "api"))


@pytest.fixture(scope="session")
def project_root():
    """Retourne le chemin racine du projet"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def models_dir():
    """Retourne le chemin du dossier models"""
    return PROJECT_ROOT / "models"


@pytest.fixture(scope="session")
def sample_cv_python():
    """CV exemple pour Python Developer"""
    return """
    Senior Python Developer with 5+ years of experience in software development.

    Skills:
    - Python, Django, FastAPI, Flask
    - Machine Learning: TensorFlow, PyTorch, scikit-learn
    - Databases: PostgreSQL, MongoDB, Redis
    - DevOps: Docker, Kubernetes, AWS

    Experience:
    - Developed ML pipelines processing 1M+ records daily
    - Built REST APIs serving 10K+ requests per minute
    - Led team of 5 developers
    """


@pytest.fixture(scope="session")
def sample_cv_java():
    """CV exemple pour Java Developer"""
    return """
    Java Developer with expertise in enterprise applications.

    Skills:
    - Java 11+, Spring Boot, Spring Cloud
    - Hibernate, JPA, JDBC
    - Microservices architecture
    - Maven, Gradle, Jenkins
    - Oracle, MySQL, PostgreSQL

    Experience:
    - Built microservices handling 5M transactions/day
    - Migrated monolith to microservices architecture
    """


@pytest.fixture(scope="session")
def sample_cv_hr():
    """CV exemple pour HR"""
    return """
    HR Manager with 8 years of experience in talent acquisition.

    Skills:
    - Recruitment and talent acquisition
    - Employee relations and engagement
    - Performance management
    - HRIS systems (Workday, SAP)
    - Training and development

    Education:
    - MBA in Human Resources Management
    """
