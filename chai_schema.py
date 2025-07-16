"""
CHAI Applied Model Card Schema Implementation
HTI-1 and OCR Compliant Model Card Generator

Based on Coalition for Healthcare AI (CHAI) standards and schema
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import re


@dataclass
class CHAIModelCard:
    """
    CHAI Applied Model Card Data Structure
    HTI-1 and OCR Compliant Model Card following CHAI schema v0.1
    """
    
    # Basic Information
    model_name: str = ""
    model_developer: str = ""
    developer_contact: str = ""
    
    # Release Information
    release_stage: str = ""
    release_date: str = ""
    release_version: str = ""
    global_availability: str = ""
    regulatory_approval: str = ""
    
    # Model Summary
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    
    # Uses and Directions
    intended_use_and_workflow: str = ""
    primary_intended_users: str = ""
    how_to_use: str = ""
    targeted_patient_population: str = ""
    cautioned_out_of_scope_settings: str = ""
    
    # Warnings
    known_risks_and_limitations: str = ""
    known_biases_or_ethical_considerations: str = ""
    clinical_risk_level: str = ""
    
    # Trust Ingredients - AI System Facts
    outcomes_and_outputs: str = ""
    model_type: str = ""
    foundation_models: str = ""
    input_data_source: str = ""
    output_and_input_data_types: str = ""
    development_data_characterization: str = ""
    bias_mitigation_approaches: str = ""
    ongoing_maintenance: str = ""
    security: str = ""
    transparency: str = ""
    
    # Transparency Information
    funding_source: str = ""
    third_party_information: str = ""
    stakeholders_consulted: str = ""
    
    # Key Metrics
    usefulness_usability_efficacy: Dict[str, str] = field(default_factory=dict)
    fairness_equity: Dict[str, str] = field(default_factory=dict)
    safety_reliability: Dict[str, str] = field(default_factory=dict)
    
    # Resources
    evaluation_references: str = ""
    clinical_trial: str = ""
    peer_reviewed_publications: str = ""
    reimbursement_status: str = ""
    patient_consent_or_disclosure: str = ""
    
    # Bibliography
    bibliography: str = ""
    
    # Metadata
    sources_extracted_from: List[str] = field(default_factory=list)
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    extraction_summary: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default metric structures"""
        if not self.usefulness_usability_efficacy:
            self.usefulness_usability_efficacy = {
                'metric_goal': '',
                'result': '',
                'interpretation': '',
                'test_type': '',
                'testing_data_description': '',
                'validation_process_and_justification': ''
            }
        
        if not self.fairness_equity:
            self.fairness_equity = {
                'metric_goal': '',
                'result': '',
                'interpretation': '',
                'test_type': '',
                'testing_data_description': '',
                'validation_process_and_justification': ''
            }
        
        if not self.safety_reliability:
            self.safety_reliability = {
                'metric_goal': '',
                'result': '',
                'interpretation': '',
                'test_type': '',
                'testing_data_description': '',
                'validation_process_and_justification': ''
            }
    
    def to_xml(self) -> str:
        """Export model card to CHAI XML format"""
        root = ET.Element('AppliedModelCard')
        root.set('xmlns', 'https://mc.chai.org/v0.1/schema.xsd')
        
        # Basic Info
        basic_info = ET.SubElement(root, 'BasicInfo')
        ET.SubElement(basic_info, 'ModelName').text = self.model_name
        ET.SubElement(basic_info, 'ModelDeveloper').text = self.model_developer
        ET.SubElement(basic_info, 'DeveloperContact').text = self.developer_contact
        
        # Release Info
        release_info = ET.SubElement(root, 'ReleaseInfo')
        ET.SubElement(release_info, 'ReleaseStage').text = self.release_stage
        ET.SubElement(release_info, 'ReleaseDate').text = self.release_date
        ET.SubElement(release_info, 'ReleaseVersion').text = self.release_version
        ET.SubElement(release_info, 'GlobalAvailability').text = self.global_availability
        if self.regulatory_approval:
            ET.SubElement(release_info, 'RegulatoryApproval').text = self.regulatory_approval
        
        # Model Summary
        model_summary = ET.SubElement(root, 'ModelSummary')
        ET.SubElement(model_summary, 'Summary').text = self.summary
        keywords_elem = ET.SubElement(model_summary, 'Keywords')
        for keyword in self.keywords:
            ET.SubElement(keywords_elem, 'Keyword').text = keyword
        
        # Uses and Directions
        uses_directions = ET.SubElement(root, 'UsesAndDirections')
        ET.SubElement(uses_directions, 'IntendedUseAndWorkflow').text = self.intended_use_and_workflow
        ET.SubElement(uses_directions, 'PrimaryIntendedUsers').text = self.primary_intended_users
        ET.SubElement(uses_directions, 'HowToUse').text = self.how_to_use
        ET.SubElement(uses_directions, 'TargetedPatientPopulation').text = self.targeted_patient_population
        ET.SubElement(uses_directions, 'CautionedOutOfScopeSettings').text = self.cautioned_out_of_scope_settings
        
        # Warnings
        warnings = ET.SubElement(root, 'Warnings')
        ET.SubElement(warnings, 'KnownRisksAndLimitations').text = self.known_risks_and_limitations
        ET.SubElement(warnings, 'KnownBiasesOrEthicalConsiderations').text = self.known_biases_or_ethical_considerations
        ET.SubElement(warnings, 'ClinicalRiskLevel').text = self.clinical_risk_level
        
        # Trust Ingredients
        trust_ingredients = ET.SubElement(root, 'TrustIngredients')
        
        # AI System Facts
        ai_facts = ET.SubElement(trust_ingredients, 'AISystemFacts')
        ET.SubElement(ai_facts, 'OutcomesAndOutputs').text = self.outcomes_and_outputs
        ET.SubElement(ai_facts, 'ModelType').text = self.model_type
        if self.foundation_models:
            ET.SubElement(ai_facts, 'FoundationModels').text = self.foundation_models
        ET.SubElement(ai_facts, 'InputDataSource').text = self.input_data_source
        ET.SubElement(ai_facts, 'OutputAndInputDataTypes').text = self.output_and_input_data_types
        ET.SubElement(ai_facts, 'DevelopmentDataCharacterization').text = self.development_data_characterization
        ET.SubElement(ai_facts, 'BiasMitigationApproaches').text = self.bias_mitigation_approaches
        ET.SubElement(ai_facts, 'OngoingMaintenance').text = self.ongoing_maintenance
        if self.security:
            ET.SubElement(ai_facts, 'Security').text = self.security
        if self.transparency:
            ET.SubElement(ai_facts, 'Transparency').text = self.transparency
        
        # Transparency Information
        transparency_info = ET.SubElement(trust_ingredients, 'TransparencyInformation')
        ET.SubElement(transparency_info, 'FundingSource').text = self.funding_source
        if self.third_party_information:
            ET.SubElement(transparency_info, 'ThirdPartyInformation').text = self.third_party_information
        ET.SubElement(transparency_info, 'StakeholdersConsulted').text = self.stakeholders_consulted
        
        # Key Metrics
        key_metrics = ET.SubElement(root, 'KeyMetrics')
        
        # Usefulness Metrics
        usefulness = ET.SubElement(key_metrics, 'UsefulnessUsabilityEfficacy')
        for key, value in self.usefulness_usability_efficacy.items():
            elem_name = ''.join(word.capitalize() for word in key.split('_'))
            ET.SubElement(usefulness, elem_name).text = value
        
        # Fairness Metrics
        fairness = ET.SubElement(key_metrics, 'FairnessEquity')
        for key, value in self.fairness_equity.items():
            elem_name = ''.join(word.capitalize() for word in key.split('_'))
            ET.SubElement(fairness, elem_name).text = value
        
        # Safety Metrics
        safety = ET.SubElement(key_metrics, 'SafetyReliability')
        for key, value in self.safety_reliability.items():
            elem_name = ''.join(word.capitalize() for word in key.split('_'))
            ET.SubElement(safety, elem_name).text = value
        
        # Resources
        resources = ET.SubElement(root, 'Resources')
        if self.evaluation_references:
            ET.SubElement(resources, 'EvaluationReferences').text = self.evaluation_references
        if self.clinical_trial:
            ET.SubElement(resources, 'ClinicalTrial').text = self.clinical_trial
        ET.SubElement(resources, 'PeerReviewedPublications').text = self.peer_reviewed_publications
        if self.reimbursement_status:
            ET.SubElement(resources, 'ReimbursementStatus').text = self.reimbursement_status
        ET.SubElement(resources, 'PatientConsentOrDisclosure').text = self.patient_consent_or_disclosure
        
        # Bibliography
        ET.SubElement(root, 'Bibliography').text = self.bibliography
        
        return ET.tostring(root, encoding='unicode')
    
    def to_json(self) -> str:
        """Export model card to JSON format"""
        return json.dumps(self.__dict__, indent=2, default=str)
    
    def to_markdown(self) -> str:
        """Export model card to Markdown format"""
        md = f"""# {self.model_name} - CHAI Applied Model Card

## Basic Information
- **Model Name**: {self.model_name}
- **Developer**: {self.model_developer}
- **Contact**: {self.developer_contact}

## Release Information
- **Stage**: {self.release_stage}
- **Date**: {self.release_date}
- **Version**: {self.release_version}
- **Availability**: {self.global_availability}
- **Regulatory**: {self.regulatory_approval}

## Model Summary
{self.summary}

**Keywords**: {', '.join(self.keywords)}

## Uses & Directions
### Intended Use and Workflow
{self.intended_use_and_workflow}

### Primary Intended Users
{self.primary_intended_users}

### How to Use
{self.how_to_use}

### Targeted Patient Population
{self.targeted_patient_population}

### Cautioned Out-of-Scope Settings
{self.cautioned_out_of_scope_settings}

## Warnings
### Known Risks and Limitations
{self.known_risks_and_limitations}

### Known Biases or Ethical Considerations
{self.known_biases_or_ethical_considerations}

### Clinical Risk Level
{self.clinical_risk_level}

## Trust Ingredients
### AI System Facts
- **Outcomes and Outputs**: {self.outcomes_and_outputs}
- **Model Type**: {self.model_type}
- **Foundation Models**: {self.foundation_models}
- **Input Data Source**: {self.input_data_source}
- **Data Types**: {self.output_and_input_data_types}

### Development Data Characterization
{self.development_data_characterization}

### Bias Mitigation Approaches
{self.bias_mitigation_approaches}

### Ongoing Maintenance
{self.ongoing_maintenance}

### Security
{self.security}

### Transparency
{self.transparency}

## Transparency Information
- **Funding Source**: {self.funding_source}
- **Third Party Information**: {self.third_party_information}
- **Stakeholders Consulted**: {self.stakeholders_consulted}

## Key Metrics
### Usefulness, Usability & Efficacy
- **Goal**: {self.usefulness_usability_efficacy.get('metric_goal', '')}
- **Result**: {self.usefulness_usability_efficacy.get('result', '')}
- **Interpretation**: {self.usefulness_usability_efficacy.get('interpretation', '')}
- **Test Type**: {self.usefulness_usability_efficacy.get('test_type', '')}
- **Testing Data**: {self.usefulness_usability_efficacy.get('testing_data_description', '')}
- **Validation Process**: {self.usefulness_usability_efficacy.get('validation_process_and_justification', '')}

### Fairness & Equity
- **Goal**: {self.fairness_equity.get('metric_goal', '')}
- **Result**: {self.fairness_equity.get('result', '')}
- **Interpretation**: {self.fairness_equity.get('interpretation', '')}
- **Test Type**: {self.fairness_equity.get('test_type', '')}
- **Testing Data**: {self.fairness_equity.get('testing_data_description', '')}
- **Validation Process**: {self.fairness_equity.get('validation_process_and_justification', '')}

### Safety & Reliability
- **Goal**: {self.safety_reliability.get('metric_goal', '')}
- **Result**: {self.safety_reliability.get('result', '')}
- **Interpretation**: {self.safety_reliability.get('interpretation', '')}
- **Test Type**: {self.safety_reliability.get('test_type', '')}
- **Testing Data**: {self.safety_reliability.get('testing_data_description', '')}
- **Validation Process**: {self.safety_reliability.get('validation_process_and_justification', '')}

## Resources
### Evaluation References
{self.evaluation_references}

### Clinical Trial
{self.clinical_trial}

### Peer-Reviewed Publications
{self.peer_reviewed_publications}

### Reimbursement Status
{self.reimbursement_status}

### Patient Consent or Disclosure
{self.patient_consent_or_disclosure}

## Bibliography
{self.bibliography}

---
*Generated on {self.extraction_timestamp}*
*Sources: {', '.join(self.sources_extracted_from)}*
"""
        return md


def extract_metrics_from_text(text: str) -> Dict[str, str]:
    """Extract performance metrics from text content"""
    metrics = {}
    
    # Common metric patterns
    patterns = {
        'sensitivity': r'sensitivity[:\s=]*(\d+\.?\d*%?)',
        'specificity': r'specificity[:\s=]*(\d+\.?\d*%?)',
        'accuracy': r'accuracy[:\s=]*(\d+\.?\d*%?)',
        'auc': r'AUC|AUROC[:\s=]*(\d+\.?\d*)',
        'precision': r'precision[:\s=]*(\d+\.?\d*%?)',
        'recall': r'recall[:\s=]*(\d+\.?\d*%?)',
        'f1': r'F1[:\s=]*(\d+\.?\d*)'
    }
    
    for metric_name, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            metrics[metric_name] = matches[0]
    
    return metrics


def extract_clinical_info_from_text(text: str) -> Dict[str, str]:
    """Extract clinical information from text content"""
    info = {}
    
    # Look for clinical oversight mentions
    oversight_patterns = [
        r'dermatologist[s]?\s+[^.]*involved',
        r'clinical oversight[^.]*',
        r'physician[s]?\s+[^.]*validated',
        r'medical professional[s]?\s+[^.]*review'
    ]
    
    for pattern in oversight_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            info['clinical_oversight'] = matches[0]
            break
    
    # Look for FDA/regulatory approval
    regulatory_patterns = [
        r'FDA[^.]*approval',
        r'CE mark[^.]*',
        r'regulatory[^.]*approval',
        r'cleared[^.]*FDA'
    ]
    
    for pattern in regulatory_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            info['regulatory_approval'] = matches[0]
            break
    
    # Look for bias mitigation
    bias_patterns = [
        r'bias[^.]*mitigation',
        r'fairness[^.]*approach',
        r'equity[^.]*consideration',
        r'diverse[^.]*dataset'
    ]
    
    for pattern in bias_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            info['bias_mitigation'] = matches[0]
            break
    
    return info