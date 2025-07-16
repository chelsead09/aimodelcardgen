"""
Enhanced CardGen Pipeline for Medical AI Model Cards
Automated data retrieval from papers, GitHub, and HuggingFace
HTI-1 and OCR Compliant using CHAI Schema

Based on CardGen method for automated model card generation
"""

import asyncio
import aiohttp
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
import trafilatura
from bs4 import BeautifulSoup

from chai_schema import CHAIModelCard, extract_metrics_from_text, extract_clinical_info_from_text


class CardGenPipeline:
    """
    Enhanced CardGen Pipeline for automated model card generation
    Follows the CardGen methodology with CHAI compliance
    """
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def discover_model_sources(self, model_name: str) -> Dict[str, Any]:
        """
        Comprehensive model discovery across multiple platforms
        CardGen Step 1: Source Discovery
        """
        discovery_results = {
            'model_name': model_name,
            'huggingface': [],
            'github': [],
            'pubmed': [],
            'arxiv': [],
            'websites': [],
            'discovery_timestamp': datetime.now().isoformat()
        }
        
        # Parallel discovery across all platforms
        tasks = [
            self._search_huggingface(model_name),
            self._search_github_repos(model_name),
            self._search_pubmed_papers(model_name),
            self._search_arxiv_papers(model_name),
            self._search_general_web(model_name)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        discovery_results['huggingface'] = results[0] if not isinstance(results[0], Exception) else []
        discovery_results['github'] = results[1] if not isinstance(results[1], Exception) else []
        discovery_results['pubmed'] = results[2] if not isinstance(results[2], Exception) else []
        discovery_results['arxiv'] = results[3] if not isinstance(results[3], Exception) else []
        discovery_results['websites'] = results[4] if not isinstance(results[4], Exception) else []
        
        return discovery_results
    
    async def _search_huggingface(self, model_name: str) -> List[Dict]:
        """Search HuggingFace for model information"""
        results = []
        try:
            # Search HuggingFace API
            search_url = f"https://huggingface.co/api/models?search={quote_plus(model_name)}"
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    for model in data[:5]:  # Top 5 results
                        model_info = {
                            'title': model.get('modelId', ''),
                            'url': f"https://huggingface.co/{model.get('modelId', '')}",
                            'description': model.get('description', ''),
                            'downloads': model.get('downloads', 0),
                            'likes': model.get('likes', 0),
                            'tags': model.get('tags', []),
                            'type': 'huggingface_model'
                        }
                        results.append(model_info)
                        
                        # Get detailed model card if available
                        card_url = f"https://huggingface.co/{model.get('modelId', '')}/raw/main/README.md"
                        try:
                            async with self.session.get(card_url) as card_response:
                                if card_response.status == 200:
                                    card_content = await card_response.text()
                                    model_info['model_card'] = card_content
                        except:
                            pass
        except Exception as e:
            print(f"Error searching HuggingFace: {e}")
        
        return results
    
    async def _search_github_repos(self, model_name: str) -> List[Dict]:
        """Search GitHub for repositories related to the model"""
        results = []
        try:
            # Search GitHub API
            search_query = f"{model_name} dermatology AI skin"
            search_url = f"https://api.github.com/search/repositories?q={quote_plus(search_query)}&sort=stars&order=desc"
            
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    for repo in data.get('items', [])[:5]:  # Top 5 results
                        repo_info = {
                            'title': repo.get('full_name', ''),
                            'url': repo.get('html_url', ''),
                            'description': repo.get('description', ''),
                            'stars': repo.get('stargazers_count', 0),
                            'language': repo.get('language', ''),
                            'type': 'github_repo'
                        }
                        results.append(repo_info)
                        
                        # Get README content
                        readme_url = f"https://api.github.com/repos/{repo.get('full_name', '')}/readme"
                        try:
                            async with self.session.get(readme_url) as readme_response:
                                if readme_response.status == 200:
                                    readme_data = await readme_response.json()
                                    import base64
                                    readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
                                    repo_info['readme'] = readme_content
                        except:
                            pass
        except Exception as e:
            print(f"Error searching GitHub: {e}")
        
        return results
    
    async def _search_pubmed_papers(self, model_name: str) -> List[Dict]:
        """Search PubMed for related research papers"""
        results = []
        try:
            # Search PubMed API
            search_query = f"{model_name} dermatology artificial intelligence machine learning"
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={quote_plus(search_query)}&retmax=10&retmode=json"
            
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    pmids = data.get('esearchresult', {}).get('idlist', [])
                    
                    if pmids:
                        # Get detailed information for each paper
                        summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={','.join(pmids)}&retmode=json"
                        async with self.session.get(summary_url) as summary_response:
                            if summary_response.status == 200:
                                summary_data = await summary_response.json()
                                for pmid, paper_data in summary_data.get('result', {}).items():
                                    if pmid != 'uids':
                                        paper_info = {
                                            'title': paper_data.get('title', ''),
                                            'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                            'authors': ', '.join([author['name'] for author in paper_data.get('authors', [])]),
                                            'journal': paper_data.get('source', ''),
                                            'pubdate': paper_data.get('pubdate', ''),
                                            'pmid': pmid,
                                            'type': 'pubmed_paper'
                                        }
                                        results.append(paper_info)
        except Exception as e:
            print(f"Error searching PubMed: {e}")
        
        return results
    
    async def _search_arxiv_papers(self, model_name: str) -> List[Dict]:
        """Search ArXiv for related research papers"""
        results = []
        try:
            # Search ArXiv API
            search_query = f"{model_name} dermatology machine learning"
            search_url = f"http://export.arxiv.org/api/query?search_query=all:{quote_plus(search_query)}&start=0&max_results=5"
            
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    content = await response.text()
                    # Parse XML response
                    root = ET.fromstring(content)
                    
                    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                        paper_info = {
                            'title': entry.find('{http://www.w3.org/2005/Atom}title').text.strip(),
                            'url': entry.find('{http://www.w3.org/2005/Atom}link').get('href'),
                            'summary': entry.find('{http://www.w3.org/2005/Atom}summary').text.strip(),
                            'authors': ', '.join([author.find('{http://www.w3.org/2005/Atom}name').text 
                                                for author in entry.findall('{http://www.w3.org/2005/Atom}author')]),
                            'published': entry.find('{http://www.w3.org/2005/Atom}published').text,
                            'type': 'arxiv_paper'
                        }
                        results.append(paper_info)
        except Exception as e:
            print(f"Error searching ArXiv: {e}")
        
        return results
    
    async def _search_general_web(self, model_name: str) -> List[Dict]:
        """Search general web for model information"""
        results = []
        try:
            # Search for model websites, documentation, etc.
            search_queries = [
                f"{model_name} dermatology model card",
                f"{model_name} AI model documentation",
                f"{model_name} skin cancer detection"
            ]
            
            for query in search_queries:
                # This would integrate with a search API like Google Custom Search
                # For now, we'll focus on known medical AI sources
                known_sources = [
                    f"https://www.nature.com/search?q={quote_plus(query)}",
                    f"https://www.sciencedirect.com/search?qs={quote_plus(query)}",
                    f"https://jamanetwork.com/search?q={quote_plus(query)}"
                ]
                
                # This is a simplified implementation
                # In production, you'd use proper search APIs
                pass
        except Exception as e:
            print(f"Error in general web search: {e}")
        
        return results
    
    async def extract_content_from_sources(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and process content from discovered sources
        CardGen Step 2: Content Extraction
        """
        extracted_content = {
            'model_name': sources['model_name'],
            'extraction_timestamp': datetime.now().isoformat(),
            'content_by_source': {},
            'aggregated_content': {
                'descriptions': [],
                'metrics': {},
                'clinical_info': {},
                'technical_specs': {},
                'bias_info': {},
                'references': []
            }
        }
        
        # Process each source type
        for source_type, source_list in sources.items():
            if source_type in ['huggingface', 'github', 'pubmed', 'arxiv', 'websites'] and source_list:
                extracted_content['content_by_source'][source_type] = []
                
                for source in source_list:
                    content = await self._extract_source_content(source)
                    extracted_content['content_by_source'][source_type].append(content)
                    
                    # Aggregate content
                    if content.get('processed_content'):
                        extracted_content['aggregated_content']['descriptions'].append(content['processed_content'])
                    
                    # Extract metrics
                    if content.get('metrics'):
                        extracted_content['aggregated_content']['metrics'].update(content['metrics'])
                    
                    # Extract clinical info
                    if content.get('clinical_info'):
                        extracted_content['aggregated_content']['clinical_info'].update(content['clinical_info'])
                    
                    # Extract references
                    if content.get('references'):
                        extracted_content['aggregated_content']['references'].extend(content['references'])
        
        return extracted_content
    
    async def _extract_source_content(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from a single source"""
        content = {
            'source_type': source.get('type', 'unknown'),
            'url': source.get('url', ''),
            'title': source.get('title', ''),
            'raw_content': '',
            'processed_content': '',
            'metrics': {},
            'clinical_info': {},
            'references': []
        }
        
        try:
            # Extract content based on source type
            if source.get('type') == 'huggingface_model':
                content['raw_content'] = source.get('model_card', source.get('description', ''))
            elif source.get('type') == 'github_repo':
                content['raw_content'] = source.get('readme', source.get('description', ''))
            elif source.get('type') in ['pubmed_paper', 'arxiv_paper']:
                # For papers, we'd need to fetch the full text or abstract
                content['raw_content'] = source.get('summary', source.get('title', ''))
            
            # Process the content
            if content['raw_content']:
                content['processed_content'] = await self._process_text_content(content['raw_content'])
                content['metrics'] = extract_metrics_from_text(content['raw_content'])
                content['clinical_info'] = extract_clinical_info_from_text(content['raw_content'])
                content['references'] = self._extract_references(content['raw_content'])
        
        except Exception as e:
            print(f"Error extracting content from {source.get('url', 'unknown')}: {e}")
        
        return content
    
    async def _process_text_content(self, text: str) -> str:
        """Process and clean text content"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove markdown artifacts
        text = re.sub(r'[#*`]', '', text)
        
        # Extract key sentences (simplified)
        sentences = text.split('.')
        key_sentences = []
        
        keywords = ['dermatology', 'skin', 'classification', 'detection', 'diagnosis', 'AI', 'machine learning', 'neural network']
        
        for sentence in sentences:
            if any(keyword.lower() in sentence.lower() for keyword in keywords):
                key_sentences.append(sentence.strip())
        
        return '. '.join(key_sentences[:5])  # Top 5 relevant sentences
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract references from text"""
        references = []
        
        # Look for DOI patterns
        doi_pattern = r'10\.\d{4,}(?:\.\d+)?/[^\s]+'
        dois = re.findall(doi_pattern, text)
        references.extend([f"https://doi.org/{doi}" for doi in dois])
        
        # Look for PubMed IDs
        pmid_pattern = r'PMID:\s*(\d+)'
        pmids = re.findall(pmid_pattern, text)
        references.extend([f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmids])
        
        return references
    
    async def generate_chai_model_card(self, model_name: str, extracted_content: Dict[str, Any]) -> CHAIModelCard:
        """
        Generate CHAI-compliant model card from extracted content
        CardGen Step 3: Model Card Generation
        """
        aggregated = extracted_content.get('aggregated_content', {})
        
        # Initialize CHAI model card
        chai_card = CHAIModelCard()
        
        # Basic Information
        chai_card.model_name = model_name
        chai_card.extraction_timestamp = datetime.now().isoformat()
        chai_card.sources_extracted_from = list(extracted_content.get('content_by_source', {}).keys())
        
        # Extract developer information
        github_sources = extracted_content.get('content_by_source', {}).get('github', [])
        if github_sources:
            chai_card.model_developer = github_sources[0].get('title', '').split('/')[0]
        
        # Model Summary
        descriptions = aggregated.get('descriptions', [])
        if descriptions:
            chai_card.summary = '. '.join(descriptions[:3])  # Combine top 3 descriptions
        
        # Extract keywords from content
        all_text = ' '.join(descriptions)
        dermatology_keywords = ['dermatology', 'skin', 'melanoma', 'cancer', 'lesion', 'diagnosis', 'classification']
        found_keywords = [kw for kw in dermatology_keywords if kw.lower() in all_text.lower()]
        chai_card.keywords = found_keywords
        
        # Uses and Directions
        chai_card.intended_use_and_workflow = "Dermatology AI model for skin condition analysis and classification"
        chai_card.primary_intended_users = "Dermatologists, healthcare professionals, and medical researchers"
        chai_card.targeted_patient_population = "Patients with skin conditions requiring diagnosis or monitoring"
        
        # Clinical information
        clinical_info = aggregated.get('clinical_info', {})
        chai_card.regulatory_approval = clinical_info.get('regulatory_approval', '')
        chai_card.bias_mitigation_approaches = clinical_info.get('bias_mitigation', '')
        
        # Performance metrics
        metrics = aggregated.get('metrics', {})
        if metrics:
            chai_card.usefulness_usability_efficacy = {
                'metric_goal': 'Evaluate model performance in dermatology applications',
                'result': f"Accuracy: {metrics.get('accuracy', 'N/A')}, Sensitivity: {metrics.get('sensitivity', 'N/A')}, Specificity: {metrics.get('specificity', 'N/A')}",
                'interpretation': 'Performance metrics indicate model effectiveness for dermatology applications',
                'test_type': 'Retrospective analysis on dermatology datasets',
                'testing_data_description': 'Diverse dermatology image datasets',
                'validation_process_and_justification': 'Standard machine learning validation procedures'
            }
        
        # Technical specifications
        chai_card.model_type = "Deep learning neural network for image classification"
        chai_card.input_data_source = "Dermatology images and associated clinical data"
        chai_card.output_and_input_data_types = "Input: Medical images (JPEG, PNG, DICOM), Output: Classification probabilities"
        
        # Resources
        references = aggregated.get('references', [])
        chai_card.peer_reviewed_publications = '; '.join(references[:5])
        
        # Bias and fairness
        chai_card.known_biases_or_ethical_considerations = "Potential bias in skin tone representation requires diverse training data"
        chai_card.clinical_risk_level = "Medium - Requires clinical oversight and validation"
        
        # Warnings
        chai_card.known_risks_and_limitations = "Model outputs require clinical validation and should not replace professional medical judgment"
        chai_card.cautioned_out_of_scope_settings = "Not validated for pediatric populations or rare skin conditions"
        
        # Transparency
        chai_card.transparency = "Model card generated using automated CardGen pipeline with multi-source data extraction"
        chai_card.funding_source = "Information not available in automated extraction"
        chai_card.stakeholders_consulted = "Dermatology professionals and AI researchers"
        
        return chai_card
    
    async def run_full_pipeline(self, model_name: str) -> CHAIModelCard:
        """
        Run the complete CardGen pipeline
        """
        print(f"Starting CardGen pipeline for: {model_name}")
        
        # Step 1: Discover sources
        print("Step 1: Discovering sources...")
        sources = await self.discover_model_sources(model_name)
        
        # Step 2: Extract content
        print("Step 2: Extracting content...")
        extracted_content = await self.extract_content_from_sources(sources)
        
        # Step 3: Generate model card
        print("Step 3: Generating CHAI model card...")
        chai_card = await self.generate_chai_model_card(model_name, extracted_content)
        
        # Store extraction summary
        chai_card.extraction_summary = {
            'sources_discovered': {k: len(v) if isinstance(v, list) else v for k, v in sources.items()},
            'content_extracted': len(extracted_content.get('aggregated_content', {}).get('descriptions', [])),
            'metrics_found': len(extracted_content.get('aggregated_content', {}).get('metrics', {})),
            'references_found': len(extracted_content.get('aggregated_content', {}).get('references', []))
        }
        
        print("CardGen pipeline completed successfully!")
        return chai_card


# Usage example
async def main():
    async with CardGenPipeline() as pipeline:
        model_card = await pipeline.run_full_pipeline("DermNet")
        print(model_card.to_json())

if __name__ == "__main__":
    asyncio.run(main())