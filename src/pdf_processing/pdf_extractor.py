"""
src/pdf_processing/pdf_extractor.py
Extraction améliorée de texte depuis les PDF avec support de différents formats
"""

import pdfplumber
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from dataclasses import dataclass


@dataclass
class PDFExtractionResult:
    """Résultat de l'extraction PDF"""
    text: str
    pages: List[str]
    metadata: Dict
    extraction_method: str
    is_scanned: bool
    tables: List[Dict]
    confidence: float


class AdvancedPDFExtractor:
    """
    Extracteur PDF avancé avec support de multiples formats et méthodes d'extraction
    """
    
    def __init__(self):
        """Initialiser l'extracteur"""
        self.supported_methods = ['pdfplumber', 'pypdf', 'ocr']
    
    def extract_from_pdf(
        self,
        pdf_path: str,
        method: str = 'auto',
        extract_tables: bool = True,
        extract_images: bool = False,
        ocr_lang: str = 'eng'
    ) -> PDFExtractionResult:
        """
        Extraire le texte d'un PDF avec détection automatique de la meilleure méthode
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            method: Méthode d'extraction ('auto', 'pdfplumber', 'pypdf', 'ocr')
            extract_tables: Extraire les tableaux
            extract_images: Extraire les images
            ocr_lang: Langue pour l'OCR (eng, fra, etc.)
        
        Returns:
            PDFExtractionResult avec le texte et les métadonnées
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"Fichier PDF non trouvé: {pdf_path}")
        
        # Détecter si le PDF est scanné
        is_scanned = self._is_scanned_pdf(pdf_path)
        
        # Choisir la méthode automatiquement si besoin
        if method == 'auto':
            method = 'ocr' if is_scanned else 'pdfplumber'
        
        # Extraire selon la méthode choisie
        if method == 'pdfplumber':
            result = self._extract_with_pdfplumber(pdf_path, extract_tables)
        elif method == 'pypdf':
            result = self._extract_with_pypdf(pdf_path)
        elif method == 'ocr':
            result = self._extract_with_ocr(pdf_path, ocr_lang)
        else:
            raise ValueError(f"Méthode non supportée: {method}")
        
        # Enrichir avec les métadonnées
        result.metadata = self._extract_metadata(pdf_path)
        result.is_scanned = is_scanned
        
        # Post-traitement du texte
        result.text = self._post_process_text(result.text)
        
        return result
    
    def _is_scanned_pdf(self, pdf_path: Path) -> bool:
        """
        Détecter si un PDF est scanné (images) ou natif (texte)
        
        Args:
            pdf_path: Chemin du PDF
        
        Returns:
            True si scanné, False sinon
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Tester les 3 premières pages
                pages_to_check = min(3, len(pdf.pages))
                
                for i in range(pages_to_check):
                    page = pdf.pages[i]
                    text = page.extract_text()
                    
                    # Si au moins une page a du texte extractible, ce n'est pas scanné
                    if text and len(text.strip()) > 50:
                        return False
                
                # Aucune page n'a de texte extractible -> probablement scanné
                return True
        except Exception:
            # En cas d'erreur, supposer que c'est scanné pour utiliser l'OCR
            return True
    
    def _extract_with_pdfplumber(
        self,
        pdf_path: Path,
        extract_tables: bool = True
    ) -> PDFExtractionResult:
        """Extraire avec pdfplumber (meilleur pour les PDFs natifs)"""
        text_parts = []
        pages_text = []
        tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extraire le texte
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
                    text_parts.append(f"\n--- Page {i+1} ---\n{page_text}")
                
                # Extraire les tableaux
                if extract_tables:
                    page_tables = page.extract_tables()
                    for j, table in enumerate(page_tables):
                        if table:
                            tables.append({
                                'page': i + 1,
                                'table_number': j + 1,
                                'data': table
                            })
        
        full_text = "\n".join(text_parts)
        
        return PDFExtractionResult(
            text=full_text,
            pages=pages_text,
            metadata={},
            extraction_method='pdfplumber',
            is_scanned=False,
            tables=tables,
            confidence=0.95  # Haute confiance pour PDFs natifs
        )
    
    def _extract_with_pypdf(self, pdf_path: Path) -> PDFExtractionResult:
        """Extraire avec pypdf (alternative plus légère)"""
        text_parts = []
        pages_text = []
        
        reader = PdfReader(pdf_path)
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
                text_parts.append(f"\n--- Page {i+1} ---\n{page_text}")
        
        full_text = "\n".join(text_parts)
        
        return PDFExtractionResult(
            text=full_text,
            pages=pages_text,
            metadata={},
            extraction_method='pypdf',
            is_scanned=False,
            tables=[],
            confidence=0.90
        )
    
    def _extract_with_ocr(
        self,
        pdf_path: Path,
        ocr_lang: str = 'eng'
    ) -> PDFExtractionResult:
        """
        Extraire avec OCR (pour PDFs scannés)
        
        Nécessite: pip install pytesseract pdf2image
        Et Tesseract installé sur le système
        """
        try:
            # Convertir PDF en images
            images = convert_from_path(
                pdf_path,
                dpi=300,  # Haute résolution pour meilleur OCR
                fmt='jpeg'
            )
            
            text_parts = []
            pages_text = []
            total_confidence = 0
            
            for i, image in enumerate(images):
                # OCR sur l'image
                ocr_data = pytesseract.image_to_data(
                    image,
                    lang=ocr_lang,
                    output_type=pytesseract.Output.DICT
                )
                
                # Extraire le texte
                page_text = pytesseract.image_to_string(image, lang=ocr_lang)
                
                if page_text:
                    pages_text.append(page_text)
                    text_parts.append(f"\n--- Page {i+1} ---\n{page_text}")
                
                # Calculer la confiance moyenne
                confidences = [
                    int(conf) for conf in ocr_data['conf'] 
                    if conf != '-1'
                ]
                if confidences:
                    total_confidence += sum(confidences) / len(confidences)
            
            full_text = "\n".join(text_parts)
            avg_confidence = total_confidence / len(images) if images else 0
            
            return PDFExtractionResult(
                text=full_text,
                pages=pages_text,
                metadata={},
                extraction_method='ocr',
                is_scanned=True,
                tables=[],
                confidence=avg_confidence / 100  # Convertir en 0-1
            )
            
        except Exception as e:
            raise RuntimeError(
                f"Erreur lors de l'OCR: {e}. "
                "Assurez-vous que Tesseract est installé et dans le PATH."
            )
    
    def _extract_metadata(self, pdf_path: Path) -> Dict:
        """Extraire les métadonnées du PDF"""
        metadata = {}
        
        try:
            reader = PdfReader(pdf_path)
            meta = reader.metadata
            
            if meta:
                metadata = {
                    'title': meta.get('/Title', ''),
                    'author': meta.get('/Author', ''),
                    'subject': meta.get('/Subject', ''),
                    'creator': meta.get('/Creator', ''),
                    'producer': meta.get('/Producer', ''),
                    'creation_date': meta.get('/CreationDate', ''),
                    'modification_date': meta.get('/ModDate', ''),
                    'num_pages': len(reader.pages)
                }
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    def _post_process_text(self, text: str) -> str:
        """
        Post-traiter le texte extrait pour améliorer la qualité
        
        Args:
            text: Texte brut extrait
        
        Returns:
            Texte nettoyé et amélioré
        """
        if not text:
            return ""
        
        # Remplacer les sauts de ligne multiples par un seul
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Supprimer les espaces en début/fin de ligne
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Supprimer les lignes vides multiples
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Corriger les espaces multiples
        text = re.sub(r' {2,}', ' ', text)
        
        # Supprimer les caractères de contrôle problématiques
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    def extract_contact_info(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extraire les informations de contact du texte
        
        Args:
            text: Texte du CV
        
        Returns:
            Dictionnaire avec email, téléphone, LinkedIn, etc.
        """
        contact_info = {
            'email': None,
            'phone': None,
            'linkedin': None,
            'github': None,
            'website': None
        }
        
        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        # Téléphone (formats variés)
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            r'\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b'
        ]
        for pattern in phone_patterns:
            phone_match = re.search(pattern, text)
            if phone_match:
                contact_info['phone'] = phone_match.group()
                break
        
        # LinkedIn
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
        if linkedin_match:
            contact_info['linkedin'] = linkedin_match.group()
        
        # GitHub
        github_pattern = r'github\.com/[\w-]+'
        github_match = re.search(github_pattern, text, re.IGNORECASE)
        if github_match:
            contact_info['github'] = github_match.group()
        
        # Website
        website_pattern = r'https?://(?:www\.)?[\w.-]+\.[a-z]{2,}'
        website_match = re.search(website_pattern, text, re.IGNORECASE)
        if website_match:
            url = website_match.group()
            # Exclure LinkedIn et GitHub
            if 'linkedin' not in url.lower() and 'github' not in url.lower():
                contact_info['website'] = url
        
        return contact_info
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extraire les sections du CV (Éducation, Expérience, Compétences, etc.)
        
        Args:
            text: Texte du CV
        
        Returns:
            Dictionnaire avec les sections identifiées
        """
        sections = {}
        
        # Patterns pour identifier les sections courantes
        section_patterns = {
            'experience': r'(?:EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT)',
            'education': r'(?:EDUCATION|ACADEMIC|QUALIFICATIONS)',
            'skills': r'(?:SKILLS|COMPETENCIES|TECHNICAL SKILLS|EXPERTISE)',
            'certifications': r'(?:CERTIFICATIONS?|LICENSES?)',
            'languages': r'(?:LANGUAGES?)',
            'projects': r'(?:PROJECTS?|PORTFOLIO)',
            'summary': r'(?:SUMMARY|PROFILE|OBJECTIVE|ABOUT)',
        }
        
        # Diviser le texte en lignes
        lines = text.split('\n')
        current_section = None
        section_content = []
        
        for line in lines:
            line_upper = line.upper().strip()
            
            # Vérifier si c'est un titre de section
            is_section_header = False
            for section_name, pattern in section_patterns.items():
                if re.match(pattern, line_upper):
                    # Sauvegarder la section précédente
                    if current_section and section_content:
                        sections[current_section] = '\n'.join(section_content).strip()
                    
                    # Commencer une nouvelle section
                    current_section = section_name
                    section_content = []
                    is_section_header = True
                    break
            
            # Ajouter la ligne à la section courante
            if not is_section_header and current_section:
                section_content.append(line)
        
        # Sauvegarder la dernière section
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content).strip()
        
        return sections
    
    def batch_extract(
        self,
        pdf_files: List[str],
        output_dir: Optional[str] = None
    ) -> List[PDFExtractionResult]:
        """
        Extraire le texte de plusieurs PDFs en batch
        
        Args:
            pdf_files: Liste des chemins de fichiers PDF
            output_dir: Dossier pour sauvegarder les textes extraits (optionnel)
        
        Returns:
            Liste des résultats d'extraction
        """
        results = []
        
        for pdf_file in pdf_files:
            print(f" Extraction de: {pdf_file}")
            
            try:
                result = self.extract_from_pdf(pdf_file)
                results.append(result)
                
                # Sauvegarder si output_dir spécifié
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    pdf_name = Path(pdf_file).stem
                    txt_file = output_path / f"{pdf_name}.txt"
                    
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        f.write(result.text)
                    
                    print(f"   Sauvegardé: {txt_file}")
                
            except Exception as e:
                print(f"   Erreur: {e}")
                results.append(None)
        
        return results


# ============================================
# EXEMPLE D'UTILISATION
# ============================================

if __name__ == "__main__":
    # Créer l'extracteur
    extractor = AdvancedPDFExtractor()
    
    # Exemple 1: Extraction automatique
    result = extractor.extract_from_pdf(
        "data/sample_cv.pdf",
        method='auto'  # Détection automatique
    )
    
    print(f"\n RÉSULTATS D'EXTRACTION")
    print(f"{'='*70}")
    print(f"Méthode utilisée: {result.extraction_method}")
    print(f"PDF scanné: {result.is_scanned}")
    print(f"Confiance: {result.confidence:.2%}")
    print(f"Nombre de pages: {len(result.pages)}")
    print(f"Nombre de tableaux: {len(result.tables)}")
    print(f"\nTexte extrait (100 premiers caractères):")
    print(result.text[:100] + "...")
    
    # Exemple 2: Extraire les informations de contact
    contact = extractor.extract_contact_info(result.text)
    print(f"\n INFORMATIONS DE CONTACT")
    print(f"{'='*70}")
    for key, value in contact.items():
        if value:
            print(f"{key.capitalize()}: {value}")
    
    # Exemple 3: Extraire les sections
    sections = extractor.extract_sections(result.text)
    print(f"\n SECTIONS IDENTIFIÉES")
    print(f"{'='*70}")
    for section_name in sections.keys():
        print(f"  - {section_name.capitalize()}")
