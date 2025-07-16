import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
import streamlit as st
from datetime import datetime
import trafilatura


@dataclass
class ModelCardData:
    # Basic Metadata
    model_name: str
    developer_organization: str = ""
    model_version: Optional[str] = None
    release_stage: str = ""  # beta, pilot, full-version
    initial_release_date: str = ""
    last_updated_date: str = ""
    contact_info: str = ""  # Website/Phone/Email
    geographic_availability: str = ""  # USA only, global, etc.
    regulatory_approval: str = ""  # FDA approval, CE Mark, etc.
    summary: str = ""
    keywords: str = ""
    clinical_oversight: str = ""  # Were dermatologists involved?
    dataset_doi: str = ""

    # Uses & Directions
    model_description: str = ""
    intended_use: str = ""
    clinical_workflow: str = ""
    clinical_gap_addressed: str = ""
    primary_users: str = ""
    how_to_use: str = ""  # web-based, app, EHR integration
    real_world_scenarios: str = ""
    cautioned_use_cases: str = ""
    inclusion_exclusion_criteria: str = ""

    # Warnings
    developer_warnings: str = ""
    model_limitations: str = ""
    failure_modes: str = ""
    dependency_requirements: str = ""
    subgroup_analysis: Dict[str, str] = None  # Fitzpatrick, age, sex, etc.
    known_biases: str = ""
    clinical_risk_level: str = ""

    # Trust Ingredients - AI System Facts
    model_type: str = ""  # predictive/generative
    system_interactions: str = ""  # EHR, medical devices
    outcomes_outputs: str = ""
    explainability: str = ""
    foundation_models: str = ""
    input_data_source: str = ""
    output_input_data_type: str = ""
    real_world_or_synthetic: str = ""
    training_inclusion_exclusion: str = ""
    uscdi_variables_used: str = ""
    demographic_representativeness: str = ""

    # Development Data
    dataset_size: str = ""
    annotation_process: str = ""
    dataset_transparency: str = ""  # public or proprietary
    validation_dataset_type: str = ""  # real-world, synthetic, retrospective
    data_collection_timeline: str = ""
    data_collection_location: str = ""
    skin_tone_diversity: str = ""
    ethical_review: str = ""
    irb_approval: str = ""
    training_data_alignment: str = ""

    # Bias Mitigation
    bias_mitigation_approaches: str = ""
    fairness_approaches: str = ""

    # Ongoing Maintenance
    monitoring_validity: str = ""
    monitoring_fairness: str = ""
    update_process: str = ""
    risk_correction: str = ""
    monitoring_tools: str = ""
    anticipated_improvements: str = ""
    security_compliance: str = ""
    transparency_mechanisms: str = ""

    # Transparency Information
    funding_source: str = ""
    third_party_info: str = ""
    stakeholders_consulted: str = ""
    conflicts_of_interest: str = ""

    # Key Metrics - structured as separate sections
    usefulness_metrics: Dict[str, str] = None
    fairness_metrics: Dict[str, str] = None
    safety_metrics: Dict[str, str] = None

    # Resources
    evaluation_references: str = ""
    clinical_trials: str = ""
    mobile_app_info: str = ""
    data_security_standards: str = ""
    compliance_frameworks: str = ""
    accreditations: str = ""
    peer_reviewed_publications: str = ""
    reimbursement_status: str = ""
    patient_consent_required: str = ""

    # Legacy fields for backward compatibility
    performance_metrics: Dict[str, Any] = None
    training_data: str = ""
    limitations: str = ""
    bias_considerations: str = ""
    ethical_considerations: str = ""
    technical_specifications: Dict[str, Any] = None
    citations: List[str] = None
    github_repo: Optional[str] = None
    huggingface_url: Optional[str] = None
    website_url: Optional[str] = None
    sources_extracted_from: List[str] = None
    source_reliability: Dict[str, Any] = None
    extraction_summary: Dict[str, Any] = None

    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.technical_specifications is None:
            self.technical_specifications = {}
        if self.citations is None:
            self.citations = []
        if self.sources_extracted_from is None:
            self.sources_extracted_from = []
        if self.source_reliability is None:
            self.source_reliability = {}
        if self.extraction_summary is None:
            self.extraction_summary = {}
        if self.subgroup_analysis is None:
            self.subgroup_analysis = {}
        if self.usefulness_metrics is None:
            self.usefulness_metrics = {
                'goal': '',
                'result': '',
                'interpretation': '',
                'test_type': '',
                'testing_data_description': '',
                'validation_process': ''
            }
        if self.fairness_metrics is None:
            self.fairness_metrics = {
                'goal': '',
                'result': '',
                'interpretation': '',
                'test_type': '',
                'testing_data_description': '',
                'validation_process': ''
            }
        if self.safety_metrics is None:
            self.safety_metrics = {
                'goal': '',
                'result': '',
                'interpretation': '',
                'test_type': '',
                'testing_data_description': '',
                'validation_process': ''
            }


class ModelDiscovery:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def search_huggingface(self, model_name: str) -> List[Dict]:
        """Search for models on HuggingFace"""
        try:
            url = f"https://huggingface.co/api/models?search={model_name}&filter=medical"
            response = self.session.get(url)
            if response.status_code == 200:
                models = response.json()
                return [
                    m for m in models if 'dermatology' in str(m).lower()
                    or 'skin' in str(m).lower()
                ]
        except Exception as e:
            st.error(f"Error searching HuggingFace: {e}")
        return []

    def search_github(self, model_name: str) -> List[Dict]:
        """Search for repositories on GitHub"""
        try:
            url = f"https://api.github.com/search/repositories?q={model_name}+dermatology+skin+AI+machine+learning"
            response = self.session.get(url)
            if response.status_code == 200:
                return response.json().get('items', [])
        except Exception as e:
            st.error(f"Error searching GitHub: {e}")
        return []

    def search_pubmed(self, model_name: str) -> List[Dict]:
        """Search for papers on PubMed"""
        try:
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            search_url = f"{base_url}esearch.fcgi?db=pubmed&term={model_name}+dermatology+artificial+intelligence&retmode=json&retmax=10"

            response = self.session.get(search_url)
            if response.status_code == 200:
                data = response.json()
                ids = data.get('esearchresult', {}).get('idlist', [])

                if ids:
                    fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={','.join(ids)}&retmode=xml"
                    papers_response = self.session.get(fetch_url)
                    if papers_response.status_code == 200:
                        return self.parse_pubmed_xml(papers_response.text)
        except Exception as e:
            st.error(f"Error searching PubMed: {e}")
        return []

    @staticmethod
    def parse_pubmed_xml(xml_content: str) -> List[Dict]:
        """Parse PubMed XML response"""
        papers = []
        soup = BeautifulSoup(xml_content, 'xml')

        for article in soup.find_all('PubmedArticle'):
            try:
                title_elem = article.find('ArticleTitle')
                title = title_elem.get_text(
                ) if title_elem else "Unknown Title"

                abstract_elem = article.find('AbstractText')
                abstract = abstract_elem.get_text() if abstract_elem else ""

                pmid_elem = article.find('PMID')
                pmid = pmid_elem.get_text() if pmid_elem else ""

                # Extract DOI if available
                doi_elem = article.find('ELocationID', {'EIdType': 'doi'})
                doi = doi_elem.get_text() if doi_elem else ""

                # Extract journal and year
                journal_elem = article.find('Title')
                journal = journal_elem.get_text() if journal_elem else "Unknown Journal"

                year_elem = article.find('PubDate')
                year = "Unknown Year"
                if year_elem:
                    year_text = year_elem.find('Year')
                    if year_text:
                        year = year_text.get_text()

                papers.append({
                    'title': title,
                    'abstract': abstract,
                    'pmid': pmid,
                    'doi': doi,
                    'journal': journal,
                    'year': year,
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                })
            except Exception:
                # Skip malformed articles
                continue

        return papers


class WebScraper:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def scrape_website(self, url: str) -> Dict[str, str]:
        """Scrape information from a website URL using trafilatura for better text extraction"""
        try:
            # Use trafilatura for better text extraction
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                if text:
                    # Also get title and metadata using BeautifulSoup
                    soup = BeautifulSoup(downloaded, 'html.parser')
                    title = soup.find('title')
                    title_text = title.get_text().strip() if title else ""
                    
                    # Extract meta description
                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    description = meta_desc.get('content', '') if meta_desc else ""
                    
                    return {
                        'title': title_text,
                        'content': text[:5000],  # Limit content length
                        'description': description,
                        'url': url
                    }
            
            # Fallback to BeautifulSoup if trafilatura fails
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract title
                title = soup.find('title')
                title_text = title.get_text().strip() if title else ""

                # Extract main content
                content_selectors = [
                    'main', 'article', '.content', '#content', '.main-content',
                    '.description', '.readme'
                ]

                content = ""
                for selector in content_selectors:
                    elem = soup.select_one(selector)
                    if elem:
                        content = elem.get_text().strip()
                        break

                if not content:
                    # Fallback to body text
                    body = soup.find('body')
                    if body:
                        content = body.get_text().strip()

                # Extract meta description
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                description = meta_desc.get('content', '') if meta_desc else ""

                return {
                    'title': title_text,
                    'content': content[:5000],  # Limit content length
                    'description': description,
                    'url': url
                }
        except Exception as e:
            st.error(f"Error scraping {url}: {e}")

        return {}


class ModelCardPipeline:

    def __init__(self):
        self.discovery = ModelDiscovery()
        self.scraper = WebScraper()

        # Skin tone diversity keywords for validation
        self.diversity_keywords = [
            'skin tone', 'skin color', 'melanin', 'fitzpatrick scale',
            'diverse population', 'ethnic diversity', 'racial diversity',
            'dark skin', 'light skin', 'brown skin', 'black skin',
            'asian skin', 'hispanic skin', 'latino skin',
            'skin type I', 'skin type II', 'skin type III', 
            'skin type IV', 'skin type V', 'skin type VI',
            'bias', 'fairness', 'equity', 'representation',
            'demographic parity', 'algorithmic bias'
        ]

        # Performance disparity keywords
        self.bias_indicators = [
            'performance gap', 'accuracy difference', 'sensitivity difference',
            'specificity difference', 'false positive rate', 'false negative rate',
            'underrepresented', 'bias evaluation', 'fairness metrics',
            'demographic bias', 'algorithmic fairness'
        ]

    def assess_source_reliability(self, sources: Dict) -> Dict[str, Any]:
        """Assess the reliability of information sources"""
        reliability_scores = {}
        
        for source_type, source_data in sources.items():
            if source_type == 'huggingface':
                # HuggingFace models are generally reliable for technical specs
                reliability_scores[source_type] = {
                    'score': 0.8,
                    'reasons': ['Official model repository', 'Community-driven', 'Technical documentation']
                }
            elif source_type == 'github':
                # GitHub repos vary in quality
                reliability_scores[source_type] = {
                    'score': 0.6,
                    'reasons': ['Code availability', 'Community contributions', 'May lack clinical validation']
                }
            elif source_type == 'pubmed':
                # PubMed papers are highly reliable
                reliability_scores[source_type] = {
                    'score': 0.9,
                    'reasons': ['Peer-reviewed', 'Scientific rigor', 'Clinical validation']
                }
            elif source_type == 'website':
                # Website reliability depends on domain
                url = source_data.get('url', '')
                if any(domain in url for domain in ['doi.org', 'pubmed.ncbi.nlm.nih.gov', 'arxiv.org']):
                    reliability_scores[source_type] = {
                        'score': 0.85,
                        'reasons': ['Academic source', 'Peer-reviewed content']
                    }
                elif any(domain in url for domain in ['.edu', '.gov']):
                    reliability_scores[source_type] = {
                        'score': 0.8,
                        'reasons': ['Educational/Government institution']
                    }
                else:
                    reliability_scores[source_type] = {
                        'score': 0.5,
                        'reasons': ['Commercial/Unknown source', 'Requires verification']
                    }
        
        return reliability_scores

    def extract_bias_indicators(self, content: str) -> Dict[str, Any]:
        """Extract bias and fairness indicators from content"""
        bias_analysis = {
            'diversity_mentions': [],
            'bias_indicators': [],
            'fairness_metrics': [],
            'bias_risk_level': 'unknown'
        }
        
        content_lower = content.lower()
        
        # Check for diversity mentions
        for keyword in self.diversity_keywords:
            if keyword in content_lower:
                # Extract context around the keyword
                pattern = rf'.{{0,100}}{re.escape(keyword)}.{{0,100}}'
                matches = re.finditer(pattern, content_lower, re.IGNORECASE)
                for match in matches:
                    bias_analysis['diversity_mentions'].append({
                        'keyword': keyword,
                        'context': match.group().strip()
                    })
        
        # Check for bias indicators
        for indicator in self.bias_indicators:
            if indicator in content_lower:
                pattern = rf'.{{0,100}}{re.escape(indicator)}.{{0,100}}'
                matches = re.finditer(pattern, content_lower, re.IGNORECASE)
                for match in matches:
                    bias_analysis['bias_indicators'].append({
                        'indicator': indicator,
                        'context': match.group().strip()
                    })
        
        # Assess bias risk level
        diversity_count = len(bias_analysis['diversity_mentions'])
        bias_count = len(bias_analysis['bias_indicators'])
        
        if diversity_count > 3 and bias_count > 1:
            bias_analysis['bias_risk_level'] = 'low'
        elif diversity_count > 1 or bias_count > 0:
            bias_analysis['bias_risk_level'] = 'medium'
        else:
            bias_analysis['bias_risk_level'] = 'high'
        
        return bias_analysis

    def discover_models(self, model_name: str) -> Dict[str, Any]:
        """Discover models from multiple sources"""
        st.info(f"Discovering models for: {model_name}")
        
        results = {
            'huggingface': [],
            'github': [],
            'pubmed': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Search HuggingFace
        with st.spinner("Searching HuggingFace..."):
            results['huggingface'] = self.discovery.search_huggingface(model_name)
        
        # Search GitHub
        with st.spinner("Searching GitHub..."):
            results['github'] = self.discovery.search_github(model_name)
        
        # Search PubMed
        with st.spinner("Searching PubMed..."):
            results['pubmed'] = self.discovery.search_pubmed(model_name)
        
        return results

    def analyze_model_sources(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze discovered sources for model information"""
        analysis = {
            'source_reliability': self.assess_source_reliability(sources),
            'bias_analysis': {},
            'extracted_info': {},
            'recommendations': []
        }
        
        # Analyze each source type
        for source_type, source_data in sources.items():
            if source_type == 'timestamp':
                continue
                
            if source_data:
                # Extract text content for analysis
                combined_text = ""
                if source_type == 'huggingface':
                    for model in source_data:
                        combined_text += f"{model.get('modelId', '')} {model.get('readme', '')} "
                elif source_type == 'github':
                    for repo in source_data:
                        combined_text += f"{repo.get('name', '')} {repo.get('description', '')} "
                elif source_type == 'pubmed':
                    for paper in source_data:
                        combined_text += f"{paper.get('title', '')} {paper.get('abstract', '')} "
                
                # Perform bias analysis
                analysis['bias_analysis'][source_type] = self.extract_bias_indicators(combined_text)
        
        # Generate recommendations
        if analysis['bias_analysis']:
            high_risk_sources = [
                source for source, bias_info in analysis['bias_analysis'].items()
                if bias_info.get('bias_risk_level') == 'high'
            ]
            
            if high_risk_sources:
                analysis['recommendations'].append(
                    f"High bias risk detected in {', '.join(high_risk_sources)}. "
                    "Consider seeking additional sources with diversity information."
                )
        
        return analysis

    def generate_model_card(self, model_name: str, sources: Dict[str, Any], 
                          analysis: Dict[str, Any]) -> ModelCardData:
        """Generate a comprehensive model card"""
        
        # Initialize model card with basic info
        model_card = ModelCardData(model_name=model_name)
        model_card.sources_extracted_from = list(sources.keys())
        model_card.source_reliability = analysis.get('source_reliability', {})
        model_card.extraction_summary = analysis.get('bias_analysis', {})
        
        # Extract information from sources
        if sources.get('huggingface'):
            hf_model = sources['huggingface'][0] if sources['huggingface'] else {}
            model_card.huggingface_url = f"https://huggingface.co/{hf_model.get('modelId', '')}"
            model_card.model_description = hf_model.get('readme', '')[:500]
        
        if sources.get('github'):
            gh_repo = sources['github'][0] if sources['github'] else {}
            model_card.github_repo = gh_repo.get('html_url', '')
            if not model_card.model_description:
                model_card.model_description = gh_repo.get('description', '')
        
        if sources.get('pubmed'):
            pub_papers = sources['pubmed'][:3]  # Take first 3 papers
            model_card.citations = [
                f"{paper.get('title', '')} ({paper.get('year', '')}) - {paper.get('url', '')}"
                for paper in pub_papers
            ]
            model_card.peer_reviewed_publications = '; '.join([
                paper.get('title', '') for paper in pub_papers
            ])
        
        # Set bias analysis results
        bias_summary = analysis.get('bias_analysis', {})
        model_card.known_biases = str(bias_summary)
        
        # Calculate overall bias risk
        risk_levels = [
            info.get('bias_risk_level', 'unknown') 
            for info in bias_summary.values()
        ]
        
        if 'high' in risk_levels:
            model_card.clinical_risk_level = 'high'
            model_card.developer_warnings = "High bias risk detected. Additional validation recommended."
        elif 'medium' in risk_levels:
            model_card.clinical_risk_level = 'medium'
            model_card.developer_warnings = "Medium bias risk detected. Review diversity metrics."
        else:
            model_card.clinical_risk_level = 'low'
        
        return model_card

    def export_model_card(self, model_card: ModelCardData, format_type: str = 'json') -> str:
        """Export model card in specified format"""
        if format_type == 'json':
            return json.dumps(asdict(model_card), indent=2, default=str)
        elif format_type == 'markdown':
            return self._generate_markdown_card(model_card)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _generate_markdown_card(self, model_card: ModelCardData) -> str:
        """Generate markdown format model card"""
        markdown = f"""# Model Card: {model_card.model_name}

## Basic Information
- **Developer Organization**: {model_card.developer_organization}
- **Model Version**: {model_card.model_version}
- **Release Stage**: {model_card.release_stage}
- **Last Updated**: {model_card.last_updated_date}

## Model Description
{model_card.model_description}

## Intended Use
{model_card.intended_use}

## Limitations and Warnings
{model_card.model_limitations}

## Bias Analysis
- **Clinical Risk Level**: {model_card.clinical_risk_level}
- **Known Biases**: {model_card.known_biases}
- **Developer Warnings**: {model_card.developer_warnings}

## Sources
- **GitHub Repository**: {model_card.github_repo}
- **HuggingFace URL**: {model_card.huggingface_url}
- **Website URL**: {model_card.website_url}

## Citations
{chr(10).join(f"- {citation}" for citation in model_card.citations)}

## Extraction Summary
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Sources: {', '.join(model_card.sources_extracted_from)}
"""
        return markdown
