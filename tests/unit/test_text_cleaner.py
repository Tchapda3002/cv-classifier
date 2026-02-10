"""
Tests unitaires pour le module text_cleaner
"""

import pytest
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestBasicCleaning:
    """Tests pour le nettoyage basique du texte"""

    def test_lowercase_conversion(self):
        """Test conversion en minuscules"""
        from preprocessing.text_cleaner import TextCleaner

        cleaner = TextCleaner()
        result = cleaner.clean_text("HELLO WORLD")

        assert result == result.lower()

    def test_url_removal(self):
        """Test suppression des URLs"""
        from preprocessing.text_cleaner import TextCleaner

        cleaner = TextCleaner()
        text = "Visit https://example.com for more info"
        result = cleaner.clean_text(text)

        assert "https" not in result
        assert "example.com" not in result

    def test_email_removal(self):
        """Test suppression des emails"""
        from preprocessing.text_cleaner import TextCleaner

        cleaner = TextCleaner()
        text = "Contact me at john.doe@email.com"
        result = cleaner.clean_text(text)

        assert "@" not in result
        assert "email.com" not in result

    def test_empty_string(self):
        """Test avec chaîne vide"""
        from preprocessing.text_cleaner import TextCleaner

        cleaner = TextCleaner()
        result = cleaner.clean_text("")

        assert result == ""

    def test_none_input(self):
        """Test avec None"""
        from preprocessing.text_cleaner import TextCleaner

        cleaner = TextCleaner()
        result = cleaner.clean_text(None)

        assert isinstance(result, str)


class TestCVCleaning:
    """Tests spécifiques au nettoyage de CV"""

    def test_cv_skills_preserved(self):
        """Test que les compétences sont préservées"""
        from preprocessing.text_cleaner import TextCleaner

        cleaner = TextCleaner()
        text = "Skills: Python, Java, Machine Learning, SQL"
        result = cleaner.clean_text(text)

        assert "python" in result.lower()
        assert "java" in result.lower()

    def test_cv_numbers_handling(self):
        """Test gestion des nombres (années d'expérience)"""
        from preprocessing.text_cleaner import TextCleaner

        cleaner = TextCleaner()
        text = "5 years of experience in software development"
        result = cleaner.clean_text(text)

        # Le résultat dépend de l'implémentation
        assert len(result) > 0

    def test_special_characters_removal(self):
        """Test suppression des caractères spéciaux"""
        from preprocessing.text_cleaner import TextCleaner

        cleaner = TextCleaner()
        text = "Python*** & Java!!! @#$%"
        result = cleaner.clean_text(text)

        assert "***" not in result
        assert "@#$%" not in result
