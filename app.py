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
    ["Model Discovery", "Model Analysis", "Model Card Generation", "Manual Entry", "Help"]
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
            if st.button("🏗️ Generate Model Card", type="primary"):
                # Need model name for generation
                if 'model_name' not in st.session_state:
                    st.session_state.model_name = st.text_input("Enter model name for card generation:")
                
                if st.session_state.get('model_name'):
                    with st.spinner("Generating model card..."):
                        st.session_state.current_model_card = st.session_state.pipeline.generate_model_card(
                            st.session_state.model_name,
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

elif page == "Manual Entry":
    st.header("✏️ Manual Model Card Entry")
    st.markdown("Manually create or edit a model card with custom information.")
    
    with st.form("manual_model_card"):
        st.subheader("Basic Information")
        
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.text_input("Model Name*", required=True)
            developer_org = st.text_input("Developer Organization")
            model_version = st.text_input("Model Version")
            release_stage = st.selectbox("Release Stage", ["", "beta", "pilot", "full-version"])
        
        with col2:
            initial_release = st.date_input("Initial Release Date", value=None)
            last_updated = st.date_input("Last Updated Date", value=None)
            contact_info = st.text_input("Contact Information")
            geographic_availability = st.text_input("Geographic Availability")
        
        st.subheader("Model Description")
        model_description = st.text_area("Model Description", height=100)
        intended_use = st.text_area("Intended Use", height=100)
        
        st.subheader("Bias and Risk Information")
        clinical_risk_level = st.selectbox("Clinical Risk Level", ["", "low", "medium", "high"])
        developer_warnings = st.text_area("Developer Warnings", height=100)
        known_biases = st.text_area("Known Biases", height=100)
        model_limitations = st.text_area("Model Limitations", height=100)
        
        st.subheader("Resources")
        github_repo = st.text_input("GitHub Repository URL")
        huggingface_url = st.text_input("HuggingFace URL")
        website_url = st.text_input("Website URL")
        
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
                model_description=model_description,
                intended_use=intended_use,
                clinical_risk_level=clinical_risk_level,
                developer_warnings=developer_warnings,
                known_biases=known_biases,
                model_limitations=model_limitations,
                github_repo=github_repo,
                huggingface_url=huggingface_url,
                website_url=website_url
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
    
    ### 4. Manual Entry
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
