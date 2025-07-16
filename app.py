import streamlit as st
import json
from datetime import datetime
from model_card_pipeline import ModelCardPipeline, ModelCardData
from utils import (
    display_source_analysis, 
    display_discovered_sources, 
    validate_model_name,
    format_model_card_display,
    create_download_button
)

# Configure Streamlit page
st.set_page_config(
    page_title="Medical AI Model Card Generator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = ModelCardPipeline()

if 'current_sources' not in st.session_state:
    st.session_state.current_sources = {}

if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = {}

if 'current_model_card' not in st.session_state:
    st.session_state.current_model_card = None

# Main app interface
st.title("🏥 Medical AI Model Card Generator")
st.markdown("Discover, analyze, and generate comprehensive model cards for medical/dermatology AI models with bias detection capabilities.")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Model Discovery", "Model Analysis", "Model Card Generation", "Website Analysis", "Manual Entry", "Help"]
)

if page == "Model Discovery":
    st.header("🔍 Model Discovery")
    st.markdown("Search for medical AI models across multiple platforms including HuggingFace, GitHub, and PubMed.")
    
    # Model search form
    with st.form("model_search_form"):
        model_name = st.text_input(
            "Enter Model Name or Keywords",
            placeholder="e.g., dermatology, skin cancer detection, melanoma classification",
            help="Enter the name of the model or relevant keywords for medical/dermatology AI models"
        )
        
        # Search options
        st.subheader("Search Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_huggingface = st.checkbox("Search HuggingFace", value=True)
        with col2:
            search_github = st.checkbox("Search GitHub", value=True)
        with col3:
            search_pubmed = st.checkbox("Search PubMed", value=True)
        
        search_button = st.form_submit_button("🔍 Search Models")
    
    # Handle search
    if search_button:
        if not validate_model_name(model_name):
            st.error("Please enter a valid model name (at least 2 characters, no special characters)")
        else:
            # Clear previous results
            st.session_state.current_sources = {}
            st.session_state.current_analysis = {}
            st.session_state.current_model_card = None
            
            # Perform search
            with st.spinner("Searching for models..."):
                sources = st.session_state.pipeline.discover_models(model_name)
                
                # Filter sources based on user selection
                if not search_huggingface:
                    sources.pop('huggingface', None)
                if not search_github:
                    sources.pop('github', None)
                if not search_pubmed:
                    sources.pop('pubmed', None)
                
                st.session_state.current_sources = sources
            
            st.success(f"Search completed! Found results from {len([k for k in sources.keys() if k != 'timestamp'])} sources.")
    
    # Display results
    if st.session_state.current_sources:
        st.header("📋 Search Results")
        
        # Summary
        sources = st.session_state.current_sources
        total_results = (
            len(sources.get('huggingface', [])) +
            len(sources.get('github', [])) +
            len(sources.get('pubmed', []))
        )
        
        st.info(f"Found {total_results} total results across all sources")
        
        # Display sources
        display_discovered_sources(sources)
        
        # Proceed to analysis button
        if st.button("➡️ Proceed to Analysis", type="primary"):
            st.session_state.current_analysis = st.session_state.pipeline.analyze_model_sources(sources)
            st.success("Analysis completed! Go to 'Model Analysis' page to view results.")

elif page == "Model Analysis":
    st.header("📊 Model Analysis")
    st.markdown("Analyze discovered models for bias, reliability, and other quality indicators.")
    
    if not st.session_state.current_sources:
        st.warning("No sources found. Please run Model Discovery first.")
        st.button("Go to Model Discovery", on_click=lambda: st.session_state.update({'page': 'Model Discovery'}))
    else:
        # Run analysis if not already done
        if not st.session_state.current_analysis:
            with st.spinner("Analyzing model sources..."):
                st.session_state.current_analysis = st.session_state.pipeline.analyze_model_sources(
                    st.session_state.current_sources
                )
        
        # Display analysis results
        if st.session_state.current_analysis:
            display_source_analysis(st.session_state.current_analysis)
            
            # Generate model card button
            st.subheader("Generate Model Card")
            model_name_for_card = st.text_input("Enter model name for card generation:", key="model_name_for_card")
            
            if st.button("🏗️ Generate Model Card", type="primary"):
                if not model_name_for_card:
                    st.error("Please enter a model name for card generation")
                else:
                    with st.spinner("Generating model card..."):
                        st.session_state.current_model_card = st.session_state.pipeline.generate_model_card(
                            model_name_for_card,
                            st.session_state.current_sources,
                            st.session_state.current_analysis
                        )
                    st.success("Model card generated! Go to 'Model Card Generation' page to view and download.")

elif page == "Model Card Generation":
    st.header("🏗️ Model Card Generation")
    st.markdown("View and download the generated model card.")
    
    if not st.session_state.current_model_card:
        st.warning("No model card generated yet. Please complete Model Discovery and Analysis first.")
        st.button("Go to Model Discovery", on_click=lambda: st.session_state.update({'page': 'Model Discovery'}))
    else:
        # Display model card
        format_model_card_display(st.session_state.current_model_card)
        
        # Download options
        st.header("📥 Download Options")
        col1, col2 = st.columns(2)
        
        with col1:
            create_download_button(st.session_state.current_model_card, 'json')
        
        with col2:
            create_download_button(st.session_state.current_model_card, 'markdown')

elif page == "Website Analysis":
    st.header("🌐 Website Analysis")
    st.markdown("Analyze a website URL to check model card accuracy and extract relevant information.")
    
    # Website URL input
    with st.form("website_analysis_form"):
        website_url = st.text_input(
            "Enter Website URL",
            placeholder="https://example.com/model-page",
            help="Enter the URL of a website containing model information"
        )
        
        st.subheader("Analysis Options")
        col1, col2 = st.columns(2)
        
        with col1:
            check_bias_indicators = st.checkbox("Check for bias indicators", value=True)
            check_diversity_mentions = st.checkbox("Check for diversity mentions", value=True)
        
        with col2:
            check_compliance = st.checkbox("Check CFR compliance", value=True)
            extract_metrics = st.checkbox("Extract performance metrics", value=True)
        
        analyze_button = st.form_submit_button("🔍 Analyze Website")
    
    # Handle analysis
    if analyze_button:
        if not website_url:
            st.error("Please enter a valid website URL")
        else:
            with st.spinner("Analyzing website..."):
                # Scrape website content
                scraped_data = st.session_state.pipeline.scraper.scrape_website(website_url)
                
                if not scraped_data:
                    st.error("Failed to scrape website content. Please check the URL and try again.")
                else:
                    st.success("Website content successfully extracted!")
                    
                    # Display extracted content
                    st.subheader("📄 Extracted Content")
                    with st.expander("Website Content", expanded=False):
                        st.markdown(f"**Title**: {scraped_data.get('title', 'N/A')}")
                        st.markdown(f"**Description**: {scraped_data.get('description', 'N/A')}")
                        st.markdown(f"**URL**: {scraped_data.get('url', 'N/A')}")
                        st.text_area("Content Preview", value=scraped_data.get('content', '')[:1000] + "...", height=200)
                    
                    # Perform bias analysis if requested
                    if check_bias_indicators or check_diversity_mentions:
                        st.subheader("🔍 Bias Analysis")
                        content = scraped_data.get('content', '')
                        bias_analysis = st.session_state.pipeline.extract_bias_indicators(content)
                        
                        # Display bias risk level
                        risk_level = bias_analysis.get('bias_risk_level', 'unknown')
                        if risk_level == 'high':
                            st.error(f"🔴 High Bias Risk Detected")
                        elif risk_level == 'medium':
                            st.warning(f"🟡 Medium Bias Risk Detected")
                        else:
                            st.success(f"🟢 Low Bias Risk")
                        
                        # Display diversity mentions
                        if check_diversity_mentions:
                            diversity_mentions = bias_analysis.get('diversity_mentions', [])
                            if diversity_mentions:
                                st.markdown("**Diversity Mentions Found:**")
                                for mention in diversity_mentions[:5]:  # Show first 5
                                    st.markdown(f"- **{mention['keyword']}**: {mention['context'][:150]}...")
                            else:
                                st.warning("No diversity mentions found in the content")
                        
                        # Display bias indicators
                        if check_bias_indicators:
                            bias_indicators = bias_analysis.get('bias_indicators', [])
                            if bias_indicators:
                                st.markdown("**Bias Indicators Found:**")
                                for indicator in bias_indicators[:5]:  # Show first 5
                                    st.markdown(f"- **{indicator['indicator']}**: {indicator['context'][:150]}...")
                            else:
                                st.info("No specific bias indicators found in the content")
                    
                    # Extract performance metrics if requested
                    if extract_metrics:
                        st.subheader("📊 Performance Metrics")
                        content = scraped_data.get('content', '')
                        
                        # Look for common performance metrics
                        metrics_patterns = {
                            'AUROC': r'AUC|AUROC|Area Under.*Curve',
                            'Accuracy': r'accuracy|acc',
                            'Sensitivity': r'sensitivity|recall|true positive rate',
                            'Specificity': r'specificity|true negative rate',
                            'Precision': r'precision|positive predictive value',
                            'F1 Score': r'F1|F-score|F1-score'
                        }
                        
                        found_metrics = {}
                        for metric_name, pattern in metrics_patterns.items():
                            matches = re.findall(rf'{pattern}[:\s=]*(\d+\.?\d*%?)', content, re.IGNORECASE)
                            if matches:
                                found_metrics[metric_name] = matches[:3]  # First 3 matches
                        
                        if found_metrics:
                            st.markdown("**Performance Metrics Found:**")
                            for metric, values in found_metrics.items():
                                st.markdown(f"- **{metric}**: {', '.join(values)}")
                        else:
                            st.info("No performance metrics found in the content")
                    
                    # Check CFR compliance if requested
                    if check_compliance:
                        st.subheader("📋 CFR Compliance Check")
                        content = scraped_data.get('content', '')
                        
                        # Check for compliance keywords
                        compliance_keywords = {
                            'Clinical Oversight': r'clinical oversight|dermatologist|clinician involvement',
                            'Bias Mitigation': r'bias mitigation|fairness|equity|demographic',
                            'Regulatory Approval': r'FDA|CE mark|regulatory approval',
                            'Validation': r'validation|testing|evaluation',
                            'Monitoring': r'monitoring|maintenance|updates',
                            'Transparency': r'transparency|funding|conflict of interest'
                        }
                        
                        compliance_found = {}
                        for category, pattern in compliance_keywords.items():
                            matches = re.findall(rf'([^.]*{pattern}[^.]*)', content, re.IGNORECASE)
                            if matches:
                                compliance_found[category] = matches[:2]  # First 2 matches
                        
                        if compliance_found:
                            st.markdown("**CFR Compliance Elements Found:**")
                            for category, matches in compliance_found.items():
                                st.markdown(f"**{category}:**")
                                for match in matches:
                                    st.markdown(f"  - {match.strip()[:200]}...")
                        else:
                            st.warning("Limited CFR compliance information found")
                    
                    # Generate model card from website data
                    st.subheader("🏗️ Generate Model Card from Website")
                    if st.button("Generate Model Card from Website Data", type="primary"):
                        with st.spinner("Generating model card from website data..."):
                            # Create a basic model card from scraped data
                            website_model_card = ModelCardData(
                                model_name=scraped_data.get('title', 'Unknown Model'),
                                model_description=scraped_data.get('description', ''),
                                website_url=website_url,
                                sources_extracted_from=['website'],
                                extraction_summary={'website_analysis': bias_analysis if 'bias_analysis' in locals() else {}}
                            )
                            
                            # Add extracted metrics if found
                            if 'found_metrics' in locals() and found_metrics:
                                website_model_card.auroc_score = ', '.join(found_metrics.get('AUROC', []))
                                website_model_card.accuracy_score = ', '.join(found_metrics.get('Accuracy', []))
                                website_model_card.sensitivity_score = ', '.join(found_metrics.get('Sensitivity', []))
                                website_model_card.specificity_score = ', '.join(found_metrics.get('Specificity', []))
                                website_model_card.f1_score = ', '.join(found_metrics.get('F1 Score', []))
                            
                            # Add bias analysis results
                            if 'bias_analysis' in locals():
                                website_model_card.clinical_risk_level = bias_analysis.get('bias_risk_level', 'unknown')
                                website_model_card.known_biases = str(bias_analysis.get('bias_indicators', []))
                            
                            st.session_state.current_model_card = website_model_card
                            st.success("Model card generated from website data! Go to 'Model Card Generation' page to view and download.")

elif page == "Manual Entry":
    st.header("✏️ Manual Model Card Entry")
    st.markdown("Manually create or edit a model card with custom information.")
    
    with st.form("manual_model_card"):
        st.subheader("1. Metadata [45 CFR 170.315 (b)(11)(iv)(B)(1)(i)]")
        
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.text_input("Model Name*")
            developer_org = st.text_input("Developer Organization")
            model_version = st.text_input("Model Version")
            release_stage = st.selectbox("Release Stage", ["", "beta", "pilot", "full-version"])
            contact_info = st.text_input("Contact Information")
            regulatory_approval = st.text_input("Regulatory Approval (FDA, CE Mark, etc.)")
        
        with col2:
            initial_release = st.date_input("Initial Release Date", value=None)
            last_updated = st.date_input("Last Updated Date", value=None)
            geographic_availability = st.text_input("Geographic Availability")
            clinical_oversight = st.text_input("Clinical Oversight")
            dataset_doi = st.text_input("Dataset DOI")
            
        summary = st.text_area("Summary", height=100)
        keywords = st.text_input("Keywords (comma-separated)")
        
        st.subheader("2. Uses & Directions")
        model_description = st.text_area("Model Description", height=100)
        intended_use = st.text_area("Intended Use", height=100)
        clinical_workflow = st.text_area("Clinical Workflow", height=100)
        
        col1, col2 = st.columns(2)
        with col1:
            primary_users = st.text_input("Primary Users")
            how_to_use = st.text_input("How to Use")
            targeted_patient_population = st.text_area("Targeted Patient Population", height=80)
            human_oversight_required = st.text_area("Human Oversight Required", height=80)
        
        with col2:
            inform_augment_replace = st.selectbox("Inform/Augment/Replace", ["", "inform", "augment", "replace"])
            specific_use_cases = st.text_area("Specific Use Cases", height=80)
            target_user_expertise = st.text_area("Target User Expertise", height=80)
            cautioned_use_cases = st.text_area("Cautioned Use Cases", height=80)
        
        st.subheader("3. Warnings [45 CFR 170.315(b)(11)(iv)(B)(3) (i-ii)]")
        clinical_risk_level = st.selectbox("Clinical Risk Level", ["", "low", "medium", "high"])
        developer_warnings = st.text_area("Developer Warnings", height=100)
        known_biases = st.text_area("Known Biases", height=100)
        model_limitations = st.text_area("Model Limitations", height=100)
        
        col1, col2 = st.columns(2)
        with col1:
            failure_modes = st.text_area("Failure Modes", height=80)
            inappropriate_settings = st.text_area("Inappropriate Settings", height=80)
        with col2:
            contraindications = st.text_area("Contraindications", height=80)
            dependency_requirements = st.text_area("Dependency Requirements", height=80)
        
        st.subheader("4. Trust Ingredients - AI System Facts")
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Model Type", ["", "predictive", "generative"])
            system_interactions = st.text_area("System Interactions", height=80)
            outcomes_outputs = st.text_area("Outcomes/Outputs", height=80)
            explainability = st.text_area("Explainability", height=80)
        
        with col2:
            foundation_models = st.text_input("Foundation Models")
            input_data_source = st.text_input("Input Data Source")
            output_input_data_type = st.text_input("Data Type")
            real_world_or_synthetic = st.selectbox("Real-World/Synthetic", ["", "real-world", "synthetic", "mixed"])
        
        st.subheader("Development Data [45 CFR 170.315(b)(11)(iv)(B)(4) (i-iv)]")
        col1, col2 = st.columns(2)
        with col1:
            dataset_size = st.text_input("Dataset Size")
            annotation_process = st.text_area("Annotation Process", height=80)
            dataset_transparency = st.selectbox("Dataset Transparency", ["", "public", "proprietary"])
            validation_dataset_type = st.selectbox("Validation Dataset Type", ["", "real-world", "synthetic", "retrospective"])
        
        with col2:
            data_collection_timeline = st.text_input("Data Collection Timeline")
            data_collection_location = st.text_input("Data Collection Location")
            skin_tone_diversity = st.text_area("Skin Tone Diversity", height=80)
            ethical_review = st.selectbox("Ethical Review", ["", "Yes", "No"])
            irb_approval = st.selectbox("IRB Approval", ["", "Yes", "No"])
        
        st.subheader("Bias Mitigation [45 CFR 170.315(b)(11)(iv)(B)(5) (i-ii)]")
        bias_mitigation_approaches = st.text_area("Bias Mitigation Approaches", height=100)
        fairness_approaches = st.text_area("Fairness Approaches", height=100)
        
        st.subheader("Key Metrics [45 CFR 170.315(b)(11)(iv)(B)(6) & (7)]")
        col1, col2, col3 = st.columns(3)
        with col1:
            auroc_score = st.text_input("AUROC Score")
            accuracy_score = st.text_input("Accuracy Score")
        with col2:
            sensitivity_score = st.text_input("Sensitivity Score")
            specificity_score = st.text_input("Specificity Score")
        with col3:
            f1_score = st.text_input("F1 Score")
            human_ai_comparison = st.text_input("Human-AI Comparison")
        
        st.subheader("Transparency Information [45 CFR 170.315 (b)(11)(iv)(B)(1)(ii)]")
        col1, col2 = st.columns(2)
        with col1:
            funding_source = st.text_input("Funding Source")
            technical_implementation_funding = st.text_input("Technical Implementation Funding")
            stakeholders_consulted = st.text_area("Stakeholders Consulted", height=80)
        with col2:
            third_party_info = st.text_input("Third Party Information")
            conflicts_of_interest = st.text_area("Conflicts of Interest", height=80)
            patient_consent_required = st.selectbox("Patient Consent Required", ["", "Yes", "No", "Recommended"])
        
        st.subheader("Resources")
        col1, col2 = st.columns(2)
        with col1:
            github_repo = st.text_input("GitHub Repository URL")
            huggingface_url = st.text_input("HuggingFace URL")
            website_url = st.text_input("Website URL")
        with col2:
            fda_status = st.selectbox("FDA Status", ["", "cleared", "in process", "investigational"])
            peer_reviewed_publications = st.text_area("Peer Reviewed Publications", height=80)
            reimbursement_status = st.text_input("Reimbursement Status")
        
        submit_button = st.form_submit_button("Create Model Card")
    
    if submit_button:
        if not model_name:
            st.error("Model name is required")
        else:
            # Create model card from manual input
            manual_card = ModelCardData(
                model_name=model_name,
                developer_organization=developer_org,
                model_version=model_version,
                release_stage=release_stage,
                initial_release_date=str(initial_release) if initial_release else "",
                last_updated_date=str(last_updated) if last_updated else "",
                contact_info=contact_info,
                geographic_availability=geographic_availability,
                regulatory_approval=regulatory_approval,
                summary=summary,
                keywords=keywords,
                clinical_oversight=clinical_oversight,
                dataset_doi=dataset_doi,
                model_description=model_description,
                intended_use=intended_use,
                clinical_workflow=clinical_workflow,
                primary_users=primary_users,
                how_to_use=how_to_use,
                targeted_patient_population=targeted_patient_population,
                human_oversight_required=human_oversight_required,
                inform_augment_replace=inform_augment_replace,
                specific_use_cases=specific_use_cases,
                target_user_expertise=target_user_expertise,
                cautioned_use_cases=cautioned_use_cases,
                clinical_risk_level=clinical_risk_level,
                developer_warnings=developer_warnings,
                known_biases=known_biases,
                model_limitations=model_limitations,
                failure_modes=failure_modes,
                inappropriate_settings=inappropriate_settings,
                contraindications=contraindications,
                dependency_requirements=dependency_requirements,
                model_type=model_type,
                system_interactions=system_interactions,
                outcomes_outputs=outcomes_outputs,
                explainability=explainability,
                foundation_models=foundation_models,
                input_data_source=input_data_source,
                output_input_data_type=output_input_data_type,
                real_world_or_synthetic=real_world_or_synthetic,
                dataset_size=dataset_size,
                annotation_process=annotation_process,
                dataset_transparency=dataset_transparency,
                validation_dataset_type=validation_dataset_type,
                data_collection_timeline=data_collection_timeline,
                data_collection_location=data_collection_location,
                skin_tone_diversity=skin_tone_diversity,
                ethical_review=ethical_review,
                irb_approval=irb_approval,
                bias_mitigation_approaches=bias_mitigation_approaches,
                fairness_approaches=fairness_approaches,
                auroc_score=auroc_score,
                accuracy_score=accuracy_score,
                sensitivity_score=sensitivity_score,
                specificity_score=specificity_score,
                f1_score=f1_score,
                human_ai_comparison=human_ai_comparison,
                funding_source=funding_source,
                technical_implementation_funding=technical_implementation_funding,
                stakeholders_consulted=stakeholders_consulted,
                third_party_info=third_party_info,
                conflicts_of_interest=conflicts_of_interest,
                patient_consent_required=patient_consent_required,
                github_repo=github_repo,
                huggingface_url=huggingface_url,
                website_url=website_url,
                fda_status=fda_status,
                peer_reviewed_publications=peer_reviewed_publications,
                reimbursement_status=reimbursement_status
            )
            
            st.session_state.current_model_card = manual_card
            st.success("Manual model card created successfully!")
            
            # Display the created card
            format_model_card_display(manual_card)
            
            # Download options
            st.header("📥 Download Options")
            col1, col2 = st.columns(2)
            
            with col1:
                create_download_button(manual_card, 'json')
            
            with col2:
                create_download_button(manual_card, 'markdown')

elif page == "Help":
    st.header("❓ Help & Documentation")
    
    st.markdown("""
    ## How to Use the Medical AI Model Card Generator
    
    This application helps you discover, analyze, and generate comprehensive model cards for medical and dermatology AI models with built-in bias detection capabilities.
    
    ### 1. Model Discovery
    - Enter a model name or keywords related to medical/dermatology AI
    - Select which sources to search (HuggingFace, GitHub, PubMed)
    - Click "Search Models" to discover relevant models
    
    ### 2. Model Analysis
    - Review the discovered sources and their reliability scores
    - Analyze bias risk indicators and diversity mentions
    - View recommendations for improving model fairness
    
    ### 3. Model Card Generation
    - Generate a comprehensive model card based on discovered information
    - Review all sections including bias analysis and risk assessment
    - Download the model card in JSON or Markdown format
    
    ### 4. Website Analysis
    - Analyze any website URL to check model card accuracy
    - Extract performance metrics and bias indicators
    - Check CFR compliance elements
    - Generate model cards from website content
    
    ### 5. Manual Entry
    - Create model cards manually when automated discovery isn't sufficient
    - Fill in all relevant information about your model
    - Generate and download the completed model card
    
    ## Key Features
    
    ### Bias Detection
    - Automatically analyzes content for diversity and bias indicators
    - Assesses skin tone diversity mentions (important for dermatology models)
    - Identifies performance disparities and fairness concerns
    - Provides risk-level assessments (low, medium, high)
    
    ### Source Reliability
    - Evaluates the credibility of different information sources
    - Weights academic sources higher than commercial ones
    - Provides transparency about information reliability
    
    ### Comprehensive Model Cards
    - Follows medical AI model card standards
    - Includes all necessary sections for clinical deployment
    - Provides structured export formats
    
    ## Important Notes
    
    - This tool is designed specifically for medical and dermatology AI models
    - Bias detection focuses on skin tone diversity and demographic representation
    - Always verify generated information with original sources
    - Consider consulting with medical professionals for clinical deployment
    
    ## Data Sources
    
    - **HuggingFace**: Model repositories and technical documentation
    - **GitHub**: Source code and development information
    - **PubMed**: Peer-reviewed scientific publications
    - **Web Scraping**: Additional model information from official websites
    
    ## Privacy and Security
    
    - No personal data is stored or transmitted
    - All searches are performed in real-time
    - Model cards are generated locally in your browser
    """)
    
    # Technical information
    with st.expander("Technical Information"):
        st.markdown("""
        ### System Requirements
        - Internet connection for API access
        - Modern web browser
        - No additional software installation required
        
        ### API Rate Limits
        - GitHub API: 60 requests per hour (unauthenticated)
        - PubMed API: 3 requests per second
        - HuggingFace API: No explicit rate limits
        
        ### Supported Model Types
        - Image classification models
        - Skin lesion detection models
        - Dermatology diagnostic models
        - Medical imaging AI models
        """)
    
    # Contact information
    with st.expander("Support & Contact"):
        st.markdown("""
        ### Getting Help
        - Check the documentation above for common questions
        - Review error messages for specific guidance
        - Ensure all required fields are completed
        
        ### Known Limitations
        - Search results depend on API availability
        - Some sources may have access restrictions
        - Generated model cards require manual verification
        """)

# Footer
st.markdown("---")
st.markdown("🏥 Medical AI Model Card Generator - Built with Streamlit")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
