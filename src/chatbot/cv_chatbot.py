"""
CV Chatbot using HuggingFace Inference API.
Allows users to ask questions about a CV.
"""

import os
from typing import Optional, Dict, List

# Charger .env si disponible
try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass


class CVChatbot:
    """
    Chatbot for analyzing CVs and answering questions.
    Uses HuggingFace Inference API with huggingface_hub.
    """

    # Models disponibles (compatibles chat sur HF Inference)
    AVAILABLE_MODELS = {
        "llama": "meta-llama/Llama-3.2-3B-Instruct",
        "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
        "phi": "microsoft/Phi-3-mini-4k-instruct",
        "mistral": "meta-llama/Llama-3.2-3B-Instruct",  # Fallback vers Llama
    }

    def __init__(
        self,
        model_name: str = "mistral",
        hf_token: Optional[str] = None
    ):
        """
        Initialize the chatbot.

        Args:
            model_name: Name of the model to use (mistral, zephyr, qwen, phi)
            hf_token: HuggingFace API token
        """
        self.model_id = self.AVAILABLE_MODELS.get(model_name, self.AVAILABLE_MODELS["mistral"])
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        self.cv_context: Optional[str] = None
        self.conversation_history: List[Dict] = []

        # Initialiser le client HuggingFace
        self.client = None
        if self.hf_token:
            try:
                from huggingface_hub import InferenceClient
                self.client = InferenceClient(token=self.hf_token)
            except ImportError:
                print("huggingface_hub non installé. pip install huggingface_hub")

    def set_cv(self, cv_text: str) -> None:
        """
        Set the CV to analyze.

        Args:
            cv_text: The raw CV text
        """
        self.cv_context = cv_text
        self.conversation_history = []

    def ask(self, question: str, max_tokens: int = 500) -> Dict:
        """
        Ask a question about the CV.

        Args:
            question: The user's question
            max_tokens: Maximum tokens in response

        Returns:
            Dict with 'answer', 'success', and 'error' keys
        """
        if not self.cv_context:
            return {
                "success": False,
                "answer": None,
                "error": "Aucun CV n'a été chargé. Veuillez d'abord charger un CV."
            }

        if not self.client:
            return {
                "success": False,
                "answer": None,
                "error": "Client HuggingFace non initialisé. Vérifiez votre token."
            }

        # Construire les messages pour le chat
        system_prompt = """Tu es un assistant spécialisé dans l'analyse de CV.
Tu dois répondre aux questions de l'utilisateur en te basant UNIQUEMENT sur le CV fourni.
Sois concis et précis. Si l'information n'est pas dans le CV, dis-le clairement.
Réponds en français."""

        cv_section = f"""Voici le CV à analyser:

{self.cv_context[:3000]}

---
Réponds maintenant à la question de l'utilisateur."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": cv_section},
        ]

        # Ajouter l'historique (derniers 3 échanges)
        for exchange in self.conversation_history[-3:]:
            messages.append({"role": "user", "content": exchange['question']})
            messages.append({"role": "assistant", "content": exchange['answer']})

        # Ajouter la question actuelle
        messages.append({"role": "user", "content": question})

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )

            answer = response.choices[0].message.content.strip()

            # Stocker dans l'historique
            self.conversation_history.append({
                "question": question,
                "answer": answer
            })

            return {
                "success": True,
                "answer": answer,
                "error": None
            }

        except Exception as e:
            error_msg = str(e)

            # Gérer les erreurs spécifiques
            if "rate limit" in error_msg.lower():
                return {
                    "success": False,
                    "answer": None,
                    "error": "Limite de requêtes atteinte. Réessayez dans quelques secondes."
                }
            elif "loading" in error_msg.lower():
                return {
                    "success": False,
                    "answer": None,
                    "error": "Le modèle est en cours de chargement. Réessayez dans 20 secondes."
                }
            else:
                return {
                    "success": False,
                    "answer": None,
                    "error": f"Erreur: {error_msg[:200]}"
                }

    def get_suggestions(self) -> List[str]:
        """Get suggested questions based on the CV."""
        return [
            "Quelles sont les compétences principales de ce candidat ?",
            "Résume ce CV en 3 points clés.",
            "Quel est le niveau d'expérience de ce candidat ?",
            "Quelles technologies maîtrise-t-il ?",
            "Est-il adapté pour un poste de développeur ?",
            "Quels sont ses points forts ?",
            "Quelle est sa formation ?",
        ]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []


# Fallback simple chatbot (sans API externe)
class SimpleCVChatbot:
    """
    Simple rule-based chatbot as fallback.
    No external API needed.
    """

    def __init__(self):
        self.cv_context: Optional[str] = None
        self.cv_lower: Optional[str] = None

    def set_cv(self, cv_text: str) -> None:
        self.cv_context = cv_text
        self.cv_lower = cv_text.lower()

    def ask(self, question: str, **kwargs) -> Dict:
        if not self.cv_context:
            return {
                "success": False,
                "answer": None,
                "error": "Aucun CV chargé."
            }

        question_lower = question.lower()

        # Simple pattern matching
        if any(w in question_lower for w in ["compétence", "skill", "technologie", "maîtrise"]):
            answer = self._extract_skills()
        elif any(w in question_lower for w in ["expérience", "année", "ans"]):
            answer = self._extract_experience()
        elif any(w in question_lower for w in ["formation", "diplôme", "étude", "éducation"]):
            answer = self._extract_education()
        elif any(w in question_lower for w in ["résume", "résumé", "synthèse", "points clés"]):
            answer = self._summarize()
        elif any(w in question_lower for w in ["contact", "email", "téléphone"]):
            answer = self._extract_contact()
        else:
            answer = "Je ne peux répondre qu'aux questions sur les compétences, l'expérience, la formation ou un résumé du CV."

        return {
            "success": True,
            "answer": answer,
            "error": None
        }

    def _extract_skills(self) -> str:
        skills_keywords = [
            "python", "java", "javascript", "c++", "sql", "html", "css",
            "react", "angular", "vue", "node", "django", "flask", "spring",
            "docker", "kubernetes", "aws", "azure", "gcp", "git",
            "machine learning", "deep learning", "tensorflow", "pytorch",
            "excel", "powerpoint", "word", "photoshop"
        ]

        found = [s for s in skills_keywords if s in self.cv_lower]

        if found:
            return f"Compétences détectées: {', '.join(found[:15])}"
        return "Aucune compétence technique standard détectée."

    def _extract_experience(self) -> str:
        import re
        years = re.findall(r'(\d+)\s*(?:ans?|years?)', self.cv_lower)
        if years:
            max_years = max(int(y) for y in years)
            return f"Expérience mentionnée: environ {max_years} ans."
        return "Durée d'expérience non clairement indiquée dans le CV."

    def _extract_education(self) -> str:
        edu_keywords = ["master", "bachelor", "licence", "ingénieur", "doctorat",
                       "phd", "bts", "dut", "bac+", "université", "école"]
        found = [e for e in edu_keywords if e in self.cv_lower]
        if found:
            return f"Formation détectée: {', '.join(found)}"
        return "Formation non clairement indiquée."

    def _extract_contact(self) -> str:
        import re
        emails = re.findall(r'[\w\.-]+@[\w\.-]+', self.cv_context)
        if emails:
            return f"Email trouvé: {emails[0]}"
        return "Pas d'email trouvé dans le CV."

    def _summarize(self) -> str:
        lines = self.cv_context.split('\n')
        non_empty = [l.strip() for l in lines if l.strip()][:5]
        return "Premiers éléments du CV:\n- " + "\n- ".join(non_empty)

    def get_suggestions(self) -> List[str]:
        return [
            "Quelles sont les compétences ?",
            "Quelle est l'expérience ?",
            "Quelle est la formation ?",
            "Résume le CV"
        ]
