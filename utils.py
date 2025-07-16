import streamlit as st
import json
from typing import Dict, Any, List
import re
from datetime import datetime


def display_source_analysis(analysis: Dict[str, Any]):
    """Display source analysis results in Streamlit"""
    
    st.subheader("Source Reliability Analysis")
    
    reliability = analysis.get('source_reliability', {})
    if reliability:
        for source, info in reliability.items():
            score = info.get('score', 0)
            reasons = info.get('reasons', [])
            
            # Color code based on reliability score
            if score >= 0.8:
                color = "green"
            elif score >= 0.6:
                color = "orange"
            else:
                color = "red"
            
            st.markdown(f"**{source.title()}**: "
                       f"<span style='color: {color}'>{score:.1f}/1.0</span>", 
                       unsafe_allow_html=True)
            
            for reason in reasons:
                st.markdown(f"  - {reason}")
    
    st.subheader("Bias Risk Analysis")
    
    bias_analysis = analysis.get('bias_analysis', {})
    if bias_analysis:
        for source_type, bias_info in bias_analysis.items():
            risk_level = bias_info.get('bias_risk_level', 'unknown')
            
            # Color code based on risk level
            if risk_level == 'low':
                color = "green"
            elif risk_level == 'medium':
                color = "orange"
            else:
                color = "red"
            
            st.markdown(f"**{source_type.title()}**: "
                       f"<span style='color: {color}'>{risk_level.title()} Risk</span>", 
                       unsafe_allow_html=True)
            
            # Display diversity mentions
            diversity_mentions = bias_info.get('diversity_mentions', [])
            if diversity_mentions:
                st.markdown("*Diversity mentions found:*")
                for mention in diversity_mentions[:3]:  # Show first 3
                    st.markdown(f"  - {mention['keyword']}: {mention['context'][:100]}...")
            
            # Display bias indicators
            bias_indicators = bias_info.get('bias_indicators', [])
            if bias_indicators:
                st.markdown("*Bias indicators found:*")
                for indicator in bias_indicators[:3]:  # Show first 3
                    st.markdown(f"  - {indicator['indicator']}: {indicator['context'][:100]}...")
    
    # Display recommendations
    recommendations = analysis.get('recommendations', [])
    if recommendations:
        st.subheader("Recommendations")
        for rec in recommendations:
            st.warning(rec)


def display_discovered_sources(sources: Dict[str, Any]):
    """Display discovered sources in an organized way"""
    
    # HuggingFace Models
    if sources.get('huggingface'):
        st.subheader("🤗 HuggingFace Models")
        for model in sources['huggingface'][:5]:  # Show first 5
            with st.expander(f"Model: {model.get('modelId', 'Unknown')}"):
                st.markdown(f"**Downloads**: {model.get('downloads', 0)}")
                st.markdown(f"**Likes**: {model.get('likes', 0)}")
                st.markdown(f"**Tags**: {', '.join(model.get('tags', []))}")
                if model.get('readme'):
                    st.markdown("**Description**:")
                    st.text(model['readme'][:500] + "..." if len(model['readme']) > 500 else model['readme'])
    
    # GitHub Repositories
    if sources.get('github'):
        st.subheader("🐙 GitHub Repositories")
        for repo in sources['github'][:5]:  # Show first 5
            with st.expander(f"Repository: {repo.get('name', 'Unknown')}"):
                st.markdown(f"**Stars**: {repo.get('stargazers_count', 0)}")
                st.markdown(f"**Language**: {repo.get('language', 'Unknown')}")
                st.markdown(f"**URL**: {repo.get('html_url', '')}")
                if repo.get('description'):
                    st.markdown(f"**Description**: {repo['description']}")
    
    # PubMed Papers
    if sources.get('pubmed'):
        st.subheader("📚 PubMed Papers")
        for paper in sources['pubmed'][:5]:  # Show first 5
            with st.expander(f"Paper: {paper.get('title', 'Unknown')[:50]}..."):
                st.markdown(f"**Journal**: {paper.get('journal', 'Unknown')}")
                st.markdown(f"**Year**: {paper.get('year', 'Unknown')}")
                st.markdown(f"**PMID**: {paper.get('pmid', 'Unknown')}")
                if paper.get('url'):
                    st.markdown(f"**URL**: {paper['url']}")
                if paper.get('abstract'):
                    st.markdown("**Abstract**:")
                    st.text(paper['abstract'][:500] + "..." if len(paper['abstract']) > 500 else paper['abstract'])


def validate_model_name(model_name: str) -> bool:
    """Validate model name input"""
    if not model_name or len(model_name.strip()) < 2:
        return False
    
    # Check for potentially harmful characters
    if re.search(r'[<>\"\'&]', model_name):
        return False
    
    return True


def format_model_card_display(model_card):
    """Format model card for display in Streamlit"""
    
    st.header(f"Model Card: {model_card.model_name}")
    
    # 1. Metadata Section
    with st.expander("📋 1. Metadata [45 CFR 170.315 (b)(11)(iv)(B)(1)(i)]", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Developer/Organization**: {model_card.developer_organization}")
            st.markdown(f"**Version**: {model_card.model_version}")
            st.markdown(f"**Release Stage**: {model_card.release_stage}")
            st.markdown(f"**Initial Release**: {model_card.initial_release_date}")
            st.markdown(f"**Contact Info**: {model_card.contact_info}")
        with col2:
            st.markdown(f"**Last Updated**: {model_card.last_updated_date}")
            st.markdown(f"**Geographic Availability**: {model_card.geographic_availability}")
            st.markdown(f"**Regulatory Approval**: {model_card.regulatory_approval}")
            st.markdown(f"**Clinical Oversight**: {model_card.clinical_oversight}")
            st.markdown(f"**Dataset DOI**: {model_card.dataset_doi}")
        
        if model_card.summary:
            st.markdown(f"**Summary**: {model_card.summary}")
        if model_card.keywords:
            st.markdown(f"**Keywords**: {model_card.keywords}")
    
    # 2. Uses & Directions
    with st.expander("📝 2. Uses & Directions"):
        st.markdown(f"**Model Description**: {model_card.model_description}")
        st.markdown(f"**Intended Use**: {model_card.intended_use}")
        st.markdown(f"**Clinical Workflow**: {model_card.clinical_workflow}")
        st.markdown(f"**Clinical Gap Addressed**: {model_card.clinical_gap_addressed}")
        st.markdown(f"**Primary Users**: {model_card.primary_users}")
        st.markdown(f"**How to Use**: {model_card.how_to_use}")
        st.markdown(f"**Targeted Patient Population**: {model_card.targeted_patient_population}")
        st.markdown(f"**Human Oversight Required**: {model_card.human_oversight_required}")
        st.markdown(f"**Inform/Augment/Replace**: {model_card.inform_augment_replace}")
        st.markdown(f"**Specific Use Cases**: {model_card.specific_use_cases}")
        st.markdown(f"**Target User Expertise**: {model_card.target_user_expertise}")
        st.markdown(f"**Real-World Scenarios**: {model_card.real_world_scenarios}")
        st.markdown(f"**Cautioned Use Cases**: {model_card.cautioned_use_cases}")
        st.markdown(f"**Inclusion/Exclusion Criteria**: {model_card.inclusion_exclusion_criteria}")
    
    # 3. Warnings
    with st.expander("⚠️ 3. Warnings [45 CFR 170.315(b)(11)(iv)(B)(3) (i-ii)]", expanded=True):
        # Risk level indicator
        risk_level = model_card.clinical_risk_level
        if risk_level == 'high':
            st.error(f"🔴 High Clinical Risk Level")
        elif risk_level == 'medium':
            st.warning(f"🟡 Medium Clinical Risk Level")
        else:
            st.success(f"🟢 Low Clinical Risk Level")
        
        if model_card.developer_warnings:
            st.warning(f"**Developer Warnings**: {model_card.developer_warnings}")
        
        st.markdown(f"**Known Biases**: {model_card.known_biases}")
        st.markdown(f"**Model Limitations**: {model_card.model_limitations}")
        st.markdown(f"**Failure Modes**: {model_card.failure_modes}")
        st.markdown(f"**Inappropriate Settings**: {model_card.inappropriate_settings}")
        st.markdown(f"**Contraindications**: {model_card.contraindications}")
        st.markdown(f"**Dependency Requirements**: {model_card.dependency_requirements}")
        
        # Subgroup Analysis
        if model_card.subgroup_analysis and any(model_card.subgroup_analysis.values()):
            st.markdown("**Subgroup Analysis**:")
            for key, value in model_card.subgroup_analysis.items():
                if value:
                    st.markdown(f"- {key.replace('_', ' ').title()}: {value}")
    
    # 4. Trust Ingredients - AI System Facts
    with st.expander("🔧 4. Trust Ingredients - AI System Facts"):
        st.markdown(f"**Model Type**: {model_card.model_type}")
        st.markdown(f"**System Interactions**: {model_card.system_interactions}")
        st.markdown(f"**Outcomes/Outputs**: {model_card.outcomes_outputs}")
        st.markdown(f"**Explainability**: {model_card.explainability}")
        st.markdown(f"**Foundation Models**: {model_card.foundation_models}")
        st.markdown(f"**Input Data Source**: {model_card.input_data_source}")
        st.markdown(f"**Data Type**: {model_card.output_input_data_type}")
        st.markdown(f"**Real-World/Synthetic**: {model_card.real_world_or_synthetic}")
        st.markdown(f"**Training Inclusion/Exclusion**: {model_card.training_inclusion_exclusion}")
        
        # USCDI v3 Variables
        if any([model_card.uscdi_race_ethnicity, model_card.uscdi_language, model_card.uscdi_sexual_orientation]):
            st.markdown("**USCDI v3 Variables**:")
            if model_card.uscdi_race_ethnicity:
                st.markdown(f"- Race/Ethnicity: {model_card.uscdi_race_ethnicity}")
            if model_card.uscdi_language:
                st.markdown(f"- Language: {model_card.uscdi_language}")
            if model_card.uscdi_sexual_orientation:
                st.markdown(f"- Sexual Orientation: {model_card.uscdi_sexual_orientation}")
            if model_card.uscdi_gender_identity:
                st.markdown(f"- Gender Identity: {model_card.uscdi_gender_identity}")
            if model_card.uscdi_sex:
                st.markdown(f"- Sex: {model_card.uscdi_sex}")
            if model_card.uscdi_social_determinants:
                st.markdown(f"- Social Determinants: {model_card.uscdi_social_determinants}")
    
    # Development Data
    with st.expander("📊 Development Data [45 CFR 170.315(b)(11)(iv)(B)(4) (i-iv)]"):
        st.markdown(f"**Dataset Size**: {model_card.dataset_size}")
        st.markdown(f"**Annotation Process**: {model_card.annotation_process}")
        st.markdown(f"**Dataset Transparency**: {model_card.dataset_transparency}")
        st.markdown(f"**Validation Dataset Type**: {model_card.validation_dataset_type}")
        st.markdown(f"**Data Collection Timeline**: {model_card.data_collection_timeline}")
        st.markdown(f"**Data Collection Location**: {model_card.data_collection_location}")
        st.markdown(f"**Skin Tone Diversity**: {model_card.skin_tone_diversity}")
        st.markdown(f"**Ethical Review**: {model_card.ethical_review}")
        st.markdown(f"**IRB Approval**: {model_card.irb_approval}")
        st.markdown(f"**Training Data Alignment**: {model_card.training_data_alignment}")
        st.markdown(f"**External Validation**: {model_card.external_validation}")
    
    # Bias Mitigation
    with st.expander("⚖️ Bias Mitigation [45 CFR 170.315(b)(11)(iv)(B)(5) (i-ii)]"):
        st.markdown(f"**Bias Mitigation Approaches**: {model_card.bias_mitigation_approaches}")
        st.markdown(f"**Fairness Approaches**: {model_card.fairness_approaches}")
        st.markdown(f"**Bias Management Strategies**: {model_card.bias_management_strategies}")
    
    # Ongoing Maintenance
    with st.expander("🔄 Ongoing Maintenance [45 CFR 170.315(b)(11)(iv)(B)(8) & (9)]"):
        st.markdown(f"**Monitoring Validity**: {model_card.monitoring_validity}")
        st.markdown(f"**Monitoring Fairness**: {model_card.monitoring_fairness}")
        st.markdown(f"**Update Process**: {model_card.update_process}")
        st.markdown(f"**Risk Correction**: {model_card.risk_correction}")
        st.markdown(f"**Monitoring Tools**: {model_card.monitoring_tools}")
        st.markdown(f"**Anticipated Improvements**: {model_card.anticipated_improvements}")
        st.markdown(f"**Security Compliance**: {model_card.security_compliance}")
        st.markdown(f"**Transparency Mechanisms**: {model_card.transparency_mechanisms}")
    
    # 5. Key Metrics
    with st.expander("📈 5. Key Metrics [45 CFR 170.315(b)(11)(iv)(B)(6) & (7)]"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Usefulness & Efficacy**")
            if model_card.usefulness_metrics and any(model_card.usefulness_metrics.values()):
                for key, value in model_card.usefulness_metrics.items():
                    if value:
                        st.markdown(f"- {key.replace('_', ' ').title()}: {value}")
        
        with col2:
            st.markdown("**Fairness & Equity**")
            if model_card.fairness_metrics and any(model_card.fairness_metrics.values()):
                for key, value in model_card.fairness_metrics.items():
                    if value:
                        st.markdown(f"- {key.replace('_', ' ').title()}: {value}")
        
        with col3:
            st.markdown("**Safety & Reliability**")
            if model_card.safety_metrics and any(model_card.safety_metrics.values()):
                for key, value in model_card.safety_metrics.items():
                    if value:
                        st.markdown(f"- {key.replace('_', ' ').title()}: {value}")
        
        # Performance Metrics Detail
        if any([model_card.auroc_score, model_card.accuracy_score, model_card.sensitivity_score]):
            st.markdown("**Performance Metrics**:")
            if model_card.auroc_score:
                st.markdown(f"- AUROC: {model_card.auroc_score}")
            if model_card.accuracy_score:
                st.markdown(f"- Accuracy: {model_card.accuracy_score}")
            if model_card.sensitivity_score:
                st.markdown(f"- Sensitivity: {model_card.sensitivity_score}")
            if model_card.specificity_score:
                st.markdown(f"- Specificity: {model_card.specificity_score}")
            if model_card.f1_score:
                st.markdown(f"- F1 Score: {model_card.f1_score}")
            if model_card.human_ai_comparison:
                st.markdown(f"- Human-AI Comparison: {model_card.human_ai_comparison}")
    
    # Transparency Information
    with st.expander("🔍 Transparency Information [45 CFR 170.315 (b)(11)(iv)(B)(1)(ii)]"):
        st.markdown(f"**Funding Source**: {model_card.funding_source}")
        st.markdown(f"**Technical Implementation Funding**: {model_card.technical_implementation_funding}")
        st.markdown(f"**Third Party Info**: {model_card.third_party_info}")
        st.markdown(f"**Stakeholders Consulted**: {model_card.stakeholders_consulted}")
        st.markdown(f"**Patient Stakeholders**: {model_card.patient_stakeholders}")
        st.markdown(f"**Provider Stakeholders**: {model_card.provider_stakeholders}")
        st.markdown(f"**Conflicts of Interest**: {model_card.conflicts_of_interest}")
    
    # 6. Resources
    with st.expander("📚 6. Resources"):
        st.markdown(f"**Evaluation References**: {model_card.evaluation_references}")
        st.markdown(f"**Clinical Trials**: {model_card.clinical_trials}")
        st.markdown(f"**Peer Reviewed Publications**: {model_card.peer_reviewed_publications}")
        st.markdown(f"**FDA Status**: {model_card.fda_status}")
        st.markdown(f"**Reimbursement Status**: {model_card.reimbursement_status}")
        st.markdown(f"**Patient Consent Required**: {model_card.patient_consent_required}")
        st.markdown(f"**Data Security Standards**: {model_card.data_security_standards}")
        st.markdown(f"**Compliance Frameworks**: {model_card.compliance_frameworks}")
        st.markdown(f"**Accreditations**: {model_card.accreditations}")
        st.markdown(f"**Privacy/Security Protocols**: {model_card.privacy_security_protocols}")
    
    # Legacy - Resources and Links
    with st.expander("🔗 Additional Resources"):
        if model_card.github_repo:
            st.markdown(f"**GitHub Repository**: {model_card.github_repo}")
        if model_card.huggingface_url:
            st.markdown(f"**HuggingFace URL**: {model_card.huggingface_url}")
        if model_card.website_url:
            st.markdown(f"**Website**: {model_card.website_url}")
        
        if model_card.citations:
            st.markdown("**Citations**:")
            for citation in model_card.citations:
                st.markdown(f"- {citation}")
    
    # Source Information
    with st.expander("🔍 Source Information"):
        if model_card.sources_extracted_from:
            st.markdown(f"**Sources**: {', '.join(model_card.sources_extracted_from)}")
        
        if model_card.source_reliability:
            st.markdown("**Source Reliability Scores**:")
            for source, info in model_card.source_reliability.items():
                score = info.get('score', 0)
                st.markdown(f"- {source.title()}: {score:.1f}/1.0")


def download_model_card(model_card, format_type: str = 'json') -> str:
    """Prepare model card for download"""
    from model_card_pipeline import ModelCardPipeline
    
    pipeline = ModelCardPipeline()
    return pipeline.export_model_card(model_card, format_type)


def create_download_button(model_card, format_type: str = 'json'):
    """Create download button for model card"""
    try:
        content = download_model_card(model_card, format_type)
        filename = f"model_card_{model_card.model_name.replace(' ', '_')}.{format_type}"
        
        st.download_button(
            label=f"Download as {format_type.upper()}",
            data=content,
            file_name=filename,
            mime='application/json' if format_type == 'json' else 'text/markdown'
        )
    except Exception as e:
        st.error(f"Error generating download: {e}")
