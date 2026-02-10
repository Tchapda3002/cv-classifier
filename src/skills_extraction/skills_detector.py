"""
src/skills_extraction/skills_detector.py
Détection automatique de compétences techniques et recommandations de postes
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter, defaultdict
import json
from pathlib import Path


class SkillsDetector:
    """
    Détecteur automatique de compétences techniques, soft skills et expérience
    avec recommandations de postes
    """
    
    def __init__(self, skills_database_path: Optional[str] = None):
        """
        Initialiser le détecteur de compétences
        
        Args:
            skills_database_path: Chemin vers une base de compétences personnalisée
        """
        self.technical_skills = self._load_technical_skills()
        self.soft_skills = self._load_soft_skills()
        self.frameworks = self._load_frameworks()
        self.languages = self._load_programming_languages()
        self.tools = self._load_tools()
        self.certifications = self._load_certifications()
        self.job_skill_mapping = self._load_job_skill_mapping()
        
        # Charger une base personnalisée si fournie
        if skills_database_path and Path(skills_database_path).exists():
            self._load_custom_database(skills_database_path)
    
    def _load_technical_skills(self) -> Dict[str, List[str]]:
        """Charger les compétences techniques par catégorie"""
        return {
            'Machine Learning': [
                'machine learning', 'deep learning', 'neural networks',
                'computer vision', 'nlp', 'natural language processing',
                'reinforcement learning', 'supervised learning',
                'unsupervised learning', 'transfer learning', 'gan',
                'generative adversarial network', 'transformers', 'bert',
                'gpt', 'llm', 'large language model'
            ],
            'Data Science': [
                'data science', 'data analysis', 'data analytics',
                'data mining', 'statistical analysis', 'statistics',
                'data visualization', 'predictive modeling',
                'regression', 'classification', 'clustering',
                'time series', 'a/b testing', 'hypothesis testing'
            ],
            'Programming': [
                'python', 'java', 'javascript', 'typescript', 'c++',
                'c#', 'ruby', 'go', 'rust', 'scala', 'kotlin',
                'swift', 'r', 'matlab', 'php', 'sql', 'nosql'
            ],
            'Web Development': [
                'html', 'css', 'frontend', 'backend', 'full stack',
                'responsive design', 'web development', 'rest api',
                'graphql', 'microservices', 'serverless',
                'progressive web app', 'pwa', 'spa', 'ssr'
            ],
            'DevOps': [
                'devops', 'ci/cd', 'continuous integration',
                'continuous deployment', 'infrastructure as code',
                'containerization', 'orchestration', 'monitoring',
                'logging', 'cloud computing', 'site reliability'
            ],
            'Databases': [
                'database', 'sql', 'postgresql', 'mysql', 'mongodb',
                'redis', 'elasticsearch', 'cassandra', 'dynamodb',
                'oracle', 'database design', 'data modeling',
                'query optimization', 'indexing', 'sharding'
            ],
            'Cloud Platforms': [
                'aws', 'azure', 'google cloud', 'gcp', 'cloud',
                'ec2', 's3', 'lambda', 'cloud functions',
                'cloud storage', 'cloud architecture'
            ],
            'Security': [
                'cybersecurity', 'security', 'encryption',
                'authentication', 'authorization', 'penetration testing',
                'vulnerability assessment', 'security audit',
                'firewall', 'intrusion detection', 'ssl', 'tls'
            ]
        }
    
    def _load_soft_skills(self) -> List[str]:
        """Charger les soft skills"""
        return [
            'communication', 'teamwork', 'leadership', 'problem solving',
            'critical thinking', 'creativity', 'adaptability',
            'time management', 'collaboration', 'presentation',
            'negotiation', 'conflict resolution', 'emotional intelligence',
            'decision making', 'project management', 'agile', 'scrum',
            'mentoring', 'coaching', 'strategic thinking'
        ]
    
    def _load_frameworks(self) -> Dict[str, List[str]]:
        """Charger les frameworks par catégorie"""
        return {
            'ML/DL Frameworks': [
                'tensorflow', 'pytorch', 'keras', 'scikit-learn',
                'sklearn', 'xgboost', 'lightgbm', 'catboost',
                'fastai', 'hugging face', 'transformers', 'spacy',
                'nltk', 'opencv', 'pandas', 'numpy', 'scipy'
            ],
            'Web Frameworks': [
                'react', 'angular', 'vue', 'svelte', 'nextjs',
                'express', 'django', 'flask', 'fastapi', 'spring',
                'spring boot', 'laravel', 'ruby on rails', 'asp.net'
            ],
            'Mobile Frameworks': [
                'react native', 'flutter', 'ionic', 'xamarin',
                'android', 'ios', 'swift ui', 'kotlin multiplatform'
            ],
            'Testing Frameworks': [
                'jest', 'mocha', 'pytest', 'unittest', 'selenium',
                'cypress', 'junit', 'testng', 'jasmine'
            ]
        }
    
    def _load_programming_languages(self) -> List[str]:
        """Charger les langages de programmation"""
        return [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#',
            'ruby', 'go', 'rust', 'scala', 'kotlin', 'swift', 'r',
            'php', 'perl', 'matlab', 'julia', 'dart', 'elixir',
            'haskell', 'clojure', 'sql', 'html', 'css'
        ]
    
    def _load_tools(self) -> Dict[str, List[str]]:
        """Charger les outils par catégorie"""
        return {
            'Version Control': ['git', 'github', 'gitlab', 'bitbucket', 'svn'],
            'Containers': ['docker', 'kubernetes', 'k8s', 'helm', 'podman'],
            'CI/CD': [
                'jenkins', 'github actions', 'gitlab ci', 'circleci',
                'travis ci', 'azure devops', 'bamboo'
            ],
            'Monitoring': [
                'prometheus', 'grafana', 'datadog', 'new relic',
                'splunk', 'elk stack', 'elasticsearch', 'kibana', 'logstash'
            ],
            'Infrastructure': [
                'terraform', 'ansible', 'chef', 'puppet', 'vagrant',
                'cloudformation', 'pulumi'
            ],
            'Data Tools': [
                'apache spark', 'hadoop', 'kafka', 'airflow',
                'tableau', 'power bi', 'looker', 'dbt', 'snowflake'
            ]
        }
    
    def _load_certifications(self) -> List[str]:
        """Charger les certifications reconnues"""
        return [
            'aws certified', 'azure certified', 'google cloud certified',
            'pmp', 'cissp', 'ceh', 'comptia', 'ccna', 'ccnp',
            'cka', 'ckad', 'scrum master', 'product owner',
            'tensorflow certificate', 'deep learning specialization'
        ]
    
    def _load_job_skill_mapping(self) -> Dict[str, Dict]:
        """
        Mapper les compétences aux rôles professionnels
        
        Returns:
            Dictionnaire avec les compétences requises par rôle
        """
        return {
            'Data Scientist': {
                'required': [
                    'python', 'machine learning', 'statistics',
                    'data analysis', 'sql'
                ],
                'preferred': [
                    'tensorflow', 'pytorch', 'scikit-learn', 'pandas',
                    'deep learning', 'nlp', 'computer vision', 'r',
                    'tableau', 'spark'
                ],
                'soft_skills': [
                    'problem solving', 'communication', 'critical thinking'
                ]
            },
            'Machine Learning Engineer': {
                'required': [
                    'python', 'machine learning', 'deep learning',
                    'tensorflow', 'pytorch'
                ],
                'preferred': [
                    'mlops', 'kubernetes', 'docker', 'aws',
                    'model deployment', 'optimization', 'spark'
                ],
                'soft_skills': [
                    'problem solving', 'teamwork', 'communication'
                ]
            },
            'Software Engineer': {
                'required': [
                    'programming', 'data structures', 'algorithms',
                    'git', 'debugging'
                ],
                'preferred': [
                    'java', 'python', 'javascript', 'c++',
                    'sql', 'rest api', 'testing', 'ci/cd'
                ],
                'soft_skills': [
                    'collaboration', 'problem solving', 'communication'
                ]
            },
            'Full Stack Developer': {
                'required': [
                    'html', 'css', 'javascript', 'backend',
                    'database', 'rest api'
                ],
                'preferred': [
                    'react', 'angular', 'vue', 'node.js', 'express',
                    'django', 'flask', 'sql', 'nosql', 'git'
                ],
                'soft_skills': [
                    'problem solving', 'creativity', 'communication'
                ]
            },
            'DevOps Engineer': {
                'required': [
                    'linux', 'ci/cd', 'docker', 'kubernetes',
                    'cloud', 'scripting'
                ],
                'preferred': [
                    'terraform', 'ansible', 'aws', 'azure', 'gcp',
                    'jenkins', 'prometheus', 'grafana', 'python', 'bash'
                ],
                'soft_skills': [
                    'problem solving', 'automation mindset', 'teamwork'
                ]
            },
            'Data Engineer': {
                'required': [
                    'sql', 'python', 'etl', 'data pipeline',
                    'data warehouse'
                ],
                'preferred': [
                    'spark', 'airflow', 'kafka', 'hadoop',
                    'aws', 'snowflake', 'dbt', 'data modeling'
                ],
                'soft_skills': [
                    'problem solving', 'attention to detail', 'communication'
                ]
            },
            'Frontend Developer': {
                'required': [
                    'html', 'css', 'javascript', 'responsive design',
                    'frontend framework'
                ],
                'preferred': [
                    'react', 'vue', 'angular', 'typescript',
                    'webpack', 'sass', 'tailwind', 'accessibility'
                ],
                'soft_skills': [
                    'attention to detail', 'creativity', 'communication'
                ]
            },
            'Backend Developer': {
                'required': [
                    'programming', 'database', 'rest api',
                    'server-side', 'backend framework'
                ],
                'preferred': [
                    'python', 'java', 'node.js', 'sql', 'nosql',
                    'microservices', 'caching', 'security'
                ],
                'soft_skills': [
                    'problem solving', 'scalability thinking', 'teamwork'
                ]
            },
            'Product Manager': {
                'required': [
                    'product management', 'roadmap', 'stakeholder',
                    'user research', 'prioritization'
                ],
                'preferred': [
                    'agile', 'scrum', 'sql', 'analytics',
                    'a/b testing', 'jira', 'market research'
                ],
                'soft_skills': [
                    'communication', 'leadership', 'strategic thinking',
                    'decision making', 'negotiation'
                ]
            },
            'Cloud Architect': {
                'required': [
                    'cloud architecture', 'aws', 'azure', 'gcp',
                    'infrastructure design', 'security'
                ],
                'preferred': [
                    'terraform', 'kubernetes', 'serverless',
                    'microservices', 'networking', 'cost optimization'
                ],
                'soft_skills': [
                    'strategic thinking', 'communication', 'problem solving'
                ]
            }
        }
    
    def _load_custom_database(self, db_path: str):
        """Charger une base de compétences personnalisée depuis JSON"""
        with open(db_path, 'r', encoding='utf-8') as f:
            custom_db = json.load(f)
        
        # Fusionner avec la base existante
        if 'technical_skills' in custom_db:
            for category, skills in custom_db['technical_skills'].items():
                if category in self.technical_skills:
                    self.technical_skills[category].extend(skills)
                else:
                    self.technical_skills[category] = skills
        
        if 'job_skill_mapping' in custom_db:
            self.job_skill_mapping.update(custom_db['job_skill_mapping'])
    
    def extract_skills(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extraire toutes les compétences du texte
        
        Args:
            text: Texte du CV
        
        Returns:
            Dictionnaire avec les compétences par catégorie
        """
        text_lower = text.lower()
        results = {
            'technical_skills': [],
            'soft_skills': [],
            'frameworks': [],
            'tools': [],
            'languages': [],
            'certifications': []
        }
        
        # Extraire les compétences techniques
        for category, skills in self.technical_skills.items():
            for skill in skills:
                if self._find_skill(skill, text_lower):
                    results['technical_skills'].append({
                        'skill': skill,
                        'category': category,
                        'confidence': self._calculate_confidence(skill, text_lower)
                    })
        
        # Extraire les soft skills
        for skill in self.soft_skills:
            if self._find_skill(skill, text_lower):
                results['soft_skills'].append({
                    'skill': skill,
                    'confidence': self._calculate_confidence(skill, text_lower)
                })
        
        # Extraire les frameworks
        for category, frameworks in self.frameworks.items():
            for framework in frameworks:
                if self._find_skill(framework, text_lower):
                    results['frameworks'].append({
                        'framework': framework,
                        'category': category,
                        'confidence': self._calculate_confidence(framework, text_lower)
                    })
        
        # Extraire les outils
        for category, tools in self.tools.items():
            for tool in tools:
                if self._find_skill(tool, text_lower):
                    results['tools'].append({
                        'tool': tool,
                        'category': category,
                        'confidence': self._calculate_confidence(tool, text_lower)
                    })
        
        # Extraire les langages de programmation
        for lang in self.languages:
            if self._find_skill(lang, text_lower):
                results['languages'].append({
                    'language': lang,
                    'confidence': self._calculate_confidence(lang, text_lower)
                })
        
        # Extraire les certifications
        for cert in self.certifications:
            if self._find_skill(cert, text_lower):
                results['certifications'].append({
                    'certification': cert,
                    'confidence': self._calculate_confidence(cert, text_lower)
                })
        
        return results
    
    def _find_skill(self, skill: str, text: str) -> bool:
        """
        Vérifier si une compétence est présente dans le texte
        
        Args:
            skill: Compétence à rechercher
            text: Texte à analyser (déjà en minuscules)
        
        Returns:
            True si trouvée, False sinon
        """
        # Recherche avec word boundaries pour éviter les faux positifs
        pattern = r'\b' + re.escape(skill) + r'\b'
        return bool(re.search(pattern, text, re.IGNORECASE))
    
    def _calculate_confidence(self, skill: str, text: str) -> float:
        """
        Calculer un score de confiance basé sur le contexte et la fréquence
        
        Args:
            skill: Compétence trouvée
            text: Texte du CV
        
        Returns:
            Score de confiance entre 0 et 1
        """
        # Compter les occurrences
        pattern = r'\b' + re.escape(skill) + r'\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        count = len(matches)
        
        # Score de base sur la fréquence (plafonné à 1.0)
        frequency_score = min(count * 0.2, 1.0)
        
        # Bonus si dans un contexte de compétences
        context_keywords = ['skills', 'expertise', 'proficient', 'experience', 
                           'knowledge', 'familiar', 'technologies']
        
        context_score = 0
        for keyword in context_keywords:
            # Chercher la compétence dans les 100 caractères autour du mot-clé
            keyword_pattern = r'.{0,50}\b' + re.escape(keyword) + r'\b.{0,50}'
            contexts = re.findall(keyword_pattern, text, re.IGNORECASE)
            
            for context in contexts:
                if skill.lower() in context.lower():
                    context_score += 0.1
        
        # Score final (moyenne pondérée)
        final_score = min((frequency_score * 0.7 + context_score * 0.3), 1.0)
        
        return round(final_score, 2)
    
    def analyze_experience(self, text: str) -> Dict:
        """
        Analyser l'expérience professionnelle
        
        Args:
            text: Texte du CV
        
        Returns:
            Informations sur l'expérience
        """
        experience_info = {
            'years_mentioned': [],
            'estimated_experience_years': 0,
            'job_titles': [],
            'companies': [],
            'experience_level': 'Unknown'
        }
        
        # Extraire les années d'expérience mentionnées
        year_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience\s+of\s+(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s+in'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            experience_info['years_mentioned'].extend([int(m) for m in matches])
        
        # Estimer le nombre d'années d'expérience
        if experience_info['years_mentioned']:
            experience_info['estimated_experience_years'] = max(
                experience_info['years_mentioned']
            )
        
        # Déterminer le niveau d'expérience
        years = experience_info['estimated_experience_years']
        if years == 0:
            experience_info['experience_level'] = 'Entry Level / Junior'
        elif years <= 2:
            experience_info['experience_level'] = 'Junior'
        elif years <= 5:
            experience_info['experience_level'] = 'Mid-Level'
        elif years <= 10:
            experience_info['experience_level'] = 'Senior'
        else:
            experience_info['experience_level'] = 'Expert / Lead'
        
        # Extraire les titres de poste courants
        job_title_patterns = [
            r'(?:senior|junior|lead|principal|staff)?\s*(?:software|data|machine learning|ml|ai|web|backend|frontend|full[ -]?stack|devops|cloud)\s+(?:engineer|developer|scientist|architect|analyst)',
            r'(?:project|product|program)\s+manager',
            r'(?:tech|technical|engineering)\s+lead',
            r'(?:cto|ceo|cio|vp)',
            r'consultant',
            r'researcher'
        ]
        
        for pattern in job_title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            experience_info['job_titles'].extend(matches)
        
        # Dédupliquer
        experience_info['job_titles'] = list(set(experience_info['job_titles']))
        
        return experience_info
    
    def recommend_jobs(
        self,
        skills: Dict[str, List[Dict]],
        experience_info: Dict,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Recommander des postes basés sur les compétences et l'expérience
        
        Args:
            skills: Compétences extraites
            experience_info: Informations sur l'expérience
            top_n: Nombre de recommandations à retourner
        
        Returns:
            Liste des recommandations de postes avec scores de correspondance
        """
        recommendations = []
        
        # Extraire toutes les compétences sous forme de set
        user_skills = set()
        
        # Compétences techniques
        for skill_dict in skills.get('technical_skills', []):
            user_skills.add(skill_dict['skill'].lower())
        
        # Frameworks
        for framework_dict in skills.get('frameworks', []):
            user_skills.add(framework_dict['framework'].lower())
        
        # Langages
        for lang_dict in skills.get('languages', []):
            user_skills.add(lang_dict['language'].lower())
        
        # Tools
        for tool_dict in skills.get('tools', []):
            user_skills.add(tool_dict['tool'].lower())
        
        # Calculer le score de correspondance pour chaque poste
        for job_title, job_reqs in self.job_skill_mapping.items():
            # Compétences requises
            required_skills = set(s.lower() for s in job_reqs['required'])
            required_match = len(user_skills & required_skills) / len(required_skills)
            
            # Compétences préférées
            preferred_skills = set(s.lower() for s in job_reqs['preferred'])
            preferred_match = len(user_skills & preferred_skills) / len(preferred_skills) if preferred_skills else 0
            
            # Soft skills
            soft_skills_required = set(s.lower() for s in job_reqs.get('soft_skills', []))
            user_soft_skills = set(s['skill'].lower() for s in skills.get('soft_skills', []))
            soft_match = len(user_soft_skills & soft_skills_required) / len(soft_skills_required) if soft_skills_required else 0
            
            # Score global (pondéré)
            overall_score = (
                required_match * 0.5 +
                preferred_match * 0.3 +
                soft_match * 0.2
            )
            
            # Compétences manquantes
            missing_required = required_skills - user_skills
            missing_preferred = preferred_skills - user_skills
            
            recommendations.append({
                'job_title': job_title,
                'match_score': round(overall_score, 2),
                'required_skills_match': round(required_match, 2),
                'preferred_skills_match': round(preferred_match, 2),
                'soft_skills_match': round(soft_match, 2),
                'missing_required_skills': list(missing_required),
                'missing_preferred_skills': list(missing_preferred)[:5],  # Top 5
                'experience_level_fit': self._check_experience_fit(
                    job_title,
                    experience_info['estimated_experience_years']
                )
            })
        
        # Trier par score et retourner le top N
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        
        return recommendations[:top_n]
    
    def _check_experience_fit(self, job_title: str, years: int) -> str:
        """Vérifier si le niveau d'expérience correspond au poste"""
        # Mapping simplifié poste -> expérience recommandée
        experience_requirements = {
            'Junior': (0, 2),
            'Mid': (2, 5),
            'Senior': (5, 10),
            'Lead': (7, 15),
            'Principal': (10, 20),
            'Architect': (8, 20),
            'Manager': (5, 15)
        }
        
        # Déterminer le niveau du poste
        job_level = 'Mid'  # Par défaut
        for level_keyword in experience_requirements.keys():
            if level_keyword.lower() in job_title.lower():
                job_level = level_keyword
                break
        
        # Vérifier la correspondance
        min_exp, max_exp = experience_requirements.get(job_level, (0, 20))
        
        if years < min_exp:
            return 'Under-qualified'
        elif years > max_exp:
            return 'Over-qualified'
        else:
            return 'Good fit'
    
    def generate_skills_report(
        self,
        text: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Générer un rapport complet d'analyse de compétences
        
        Args:
            text: Texte du CV
            output_path: Chemin pour sauvegarder le rapport (optionnel)
        
        Returns:
            Rapport complet
        """
        # Extraire les compétences
        skills = self.extract_skills(text)
        
        # Analyser l'expérience
        experience = self.analyze_experience(text)
        
        # Recommandations de postes
        recommendations = self.recommend_jobs(skills, experience, top_n=5)
        
        # Créer le rapport
        report = {
            'skills_summary': {
                'total_technical_skills': len(skills['technical_skills']),
                'total_soft_skills': len(skills['soft_skills']),
                'total_frameworks': len(skills['frameworks']),
                'total_tools': len(skills['tools']),
                'total_languages': len(skills['languages']),
                'total_certifications': len(skills['certifications'])
            },
            'detailed_skills': skills,
            'experience_analysis': experience,
            'job_recommendations': recommendations,
            'top_strengths': self._identify_top_strengths(skills),
            'skill_gaps': self._identify_skill_gaps(skills, recommendations)
        }
        
        # Sauvegarder si demandé
        if output_path:
            # Format JSON
            if output_path.endswith('.json'):
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Format texte
            else:
                self._save_text_report(report, output_path)
            
            print(f" Rapport sauvegardé: {output_path}")
        
        return report
    
    def _identify_top_strengths(self, skills: Dict) -> List[str]:
        """Identifier les principaux atouts"""
        strengths = []
        
        # Compter les compétences par catégorie technique
        tech_categories = defaultdict(int)
        for skill in skills.get('technical_skills', []):
            tech_categories[skill['category']] += 1
        
        # Top 3 catégories
        top_categories = sorted(
            tech_categories.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for category, count in top_categories:
            strengths.append(f"Strong in {category} ({count} skills)")
        
        return strengths
    
    def _identify_skill_gaps(
        self,
        skills: Dict,
        recommendations: List[Dict]
    ) -> Dict:
        """Identifier les lacunes de compétences"""
        if not recommendations:
            return {}
        
        # Prendre le poste le mieux noté
        top_job = recommendations[0]
        
        return {
            'for_job': top_job['job_title'],
            'missing_required': top_job['missing_required_skills'],
            'missing_preferred': top_job['missing_preferred_skills']
        }
    
    def _save_text_report(self, report: Dict, output_path: str):
        """Sauvegarder le rapport en format texte"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAPPORT D'ANALYSE DE COMPÉTENCES\n")
            f.write("=" * 80 + "\n\n")
            
            # Résumé
            f.write(" RÉSUMÉ DES COMPÉTENCES\n")
            f.write("-" * 80 + "\n")
            summary = report['skills_summary']
            f.write(f"Compétences techniques: {summary['total_technical_skills']}\n")
            f.write(f"Soft skills: {summary['total_soft_skills']}\n")
            f.write(f"Frameworks: {summary['total_frameworks']}\n")
            f.write(f"Outils: {summary['total_tools']}\n")
            f.write(f"Langages: {summary['total_languages']}\n")
            f.write(f"Certifications: {summary['total_certifications']}\n\n")
            
            # Expérience
            f.write(" ANALYSE DE L'EXPÉRIENCE\n")
            f.write("-" * 80 + "\n")
            exp = report['experience_analysis']
            f.write(f"Années d'expérience: {exp['estimated_experience_years']}\n")
            f.write(f"Niveau: {exp['experience_level']}\n\n")
            
            # Recommandations
            f.write(" RECOMMANDATIONS DE POSTES (Top 5)\n")
            f.write("-" * 80 + "\n")
            for i, rec in enumerate(report['job_recommendations'], 1):
                f.write(f"\n{i}. {rec['job_title']}\n")
                f.write(f"   Score de correspondance: {rec['match_score']:.0%}\n")
                f.write(f"   Fit d'expérience: {rec['experience_level_fit']}\n")
                if rec['missing_required_skills']:
                    f.write(f"   Compétences requises manquantes: {', '.join(rec['missing_required_skills'][:3])}\n")
            
            f.write("\n" + "=" * 80 + "\n")


# ============================================
# EXEMPLE D'UTILISATION
# ============================================

if __name__ == "__main__":
    # Créer le détecteur
    detector = SkillsDetector()
    
    # Exemple de texte de CV
    sample_cv_text = """
    Senior Data Scientist with 7 years of experience in machine learning and deep learning.
    
    SKILLS:
    - Programming: Python, R, SQL
    - ML/DL: TensorFlow, PyTorch, scikit-learn, Keras
    - Data: Pandas, NumPy, Spark, Hadoop
    - Cloud: AWS (SageMaker, EC2, S3), Docker, Kubernetes
    - Tools: Git, Jupyter, MLflow
    
    EXPERIENCE:
    Led a team of 5 data scientists, developed NLP models for customer sentiment analysis.
    Implemented computer vision solutions using CNNs and transformers.
    
    SOFT SKILLS:
    Strong communication, leadership, and problem-solving skills.
    Experience with Agile and Scrum methodologies.
    """
    
    # Générer le rapport
    report = detector.generate_skills_report(
        sample_cv_text,
        output_path="outputs/reports/skills_analysis.txt"
    )
    
    print("\n RAPPORT GÉNÉRÉ")
    print("=" * 70)
    print(f"Compétences techniques trouvées: {report['skills_summary']['total_technical_skills']}")
    print(f"Soft skills trouvées: {report['skills_summary']['total_soft_skills']}")
    print(f"\nTop 3 recommandations de postes:")
    for i, rec in enumerate(report['job_recommendations'][:3], 1):
        print(f"  {i}. {rec['job_title']} - {rec['match_score']:.0%} match")
