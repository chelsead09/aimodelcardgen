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
    
    # Basic Information
    with st.expander("📋 Basic Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Developer**: {model_card.developer_organization}")
            st.markdown(f"**Version**: {model_card.model_version}")
            st.markdown(f"**Release Stage**: {model_card.release_stage}")
        with col2:
            st.markdown(f"**Last Updated**: {model_card.last_updated_date}")
            st.markdown(f"**Contact**: {model_card.contact_info}")
            st.markdown(f"**Geographic Availability**: {model_card.geographic_availability}")
    
    # Model Description
    with st.expander("📝 Model Description"):
        st.markdown(model_card.model_description)
        if model_card.intended_use:
            st.markdown(f"**Intended Use**: {model_card.intended_use}")
    
    # Bias and Risk Analysis
    with st.expander("⚠️ Bias and Risk Analysis", expanded=True):
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
        
        if model_card.known_biases:
            st.markdown(f"**Known Biases**: {model_card.known_biases}")
        
        if model_card.model_limitations:
            st.markdown(f"**Limitations**: {model_card.model_limitations}")
    
    # Performance Metrics
    with st.expander("📊 Performance Metrics"):
        if model_card.usefulness_metrics and any(model_card.usefulness_metrics.values()):
            st.markdown("**Usefulness Metrics**:")
            for key, value in model_card.usefulness_metrics.items():
                if value:
                    st.markdown(f"- {key.replace('_', ' ').title()}: {value}")
        
        if model_card.fairness_metrics and any(model_card.fairness_metrics.values()):
            st.markdown("**Fairness Metrics**:")
            for key, value in model_card.fairness_metrics.items():
                if value:
                    st.markdown(f"- {key.replace('_', ' ').title()}: {value}")
        
        if model_card.safety_metrics and any(model_card.safety_metrics.values()):
            st.markdown("**Safety Metrics**:")
            for key, value in model_card.safety_metrics.items():
                if value:
                    st.markdown(f"- {key.replace('_', ' ').title()}: {value}")
    
    # Resources and Links
    with st.expander("🔗 Resources and Links"):
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
