# Medical AI Model Card Generator

## Overview

This is a Streamlit-based web application designed to automatically discover, analyze, and generate comprehensive model cards for medical and dermatology AI models. The system focuses on bias detection capabilities and provides structured documentation for AI models used in healthcare settings.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **User Interface**: Multi-page application with sidebar navigation
- **Session Management**: Streamlit session state for maintaining user data across pages
- **Layout**: Wide layout with expandable sidebar for navigation

### Backend Architecture
- **Core Pipeline**: `ModelCardPipeline` class handles model discovery and analysis
- **Data Processing**: Asynchronous web scraping using aiohttp and BeautifulSoup
- **Content Extraction**: Trafilatura library for web content extraction
- **Data Structure**: Dataclass-based model card schema for structured data storage

### Key Components

#### Model Discovery System
- Multi-platform search across HuggingFace, GitHub, and PubMed
- Asynchronous web scraping for efficient data collection
- Source reliability scoring system
- Bias risk assessment for discovered models

#### Model Card Generation
- Structured data schema based on medical AI requirements
- Comprehensive fields covering metadata, usage directions, and warnings
- Clinical oversight tracking and regulatory approval documentation
- Subgroup analysis for bias detection (Fitzpatrick scale, age, sex demographics)

#### Analysis Engine
- Source reliability analysis with scoring system (0-1.0 scale)
- Bias risk assessment with color-coded risk levels (low/medium/high)
- Clinical risk level categorization
- Failure mode identification and documentation

## Data Flow

1. **Model Discovery**: User searches for medical AI models by name/keywords
2. **Source Aggregation**: System scrapes multiple platforms asynchronously
3. **Content Analysis**: Extracted content is analyzed for reliability and bias
4. **Model Card Generation**: Structured data is compiled into comprehensive model cards
5. **Manual Override**: Users can manually enter or edit model information
6. **Export**: Generated model cards can be downloaded in various formats

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **aiohttp**: Asynchronous HTTP client for web scraping
- **BeautifulSoup**: HTML parsing and web scraping
- **trafilatura**: Web content extraction and text processing
- **requests**: HTTP library for API calls

### Data Sources
- **HuggingFace**: Model repository platform
- **GitHub**: Code repository platform
- **PubMed**: Medical literature database

## Deployment Strategy

### Development Environment
- Python-based application suitable for local development
- Streamlit dev server for testing and development
- Session state management for user data persistence

### Production Considerations
- Streamlit Cloud deployment ready
- Asynchronous processing for handling multiple concurrent requests
- No database dependency - uses in-memory session state
- Stateless architecture suitable for cloud deployment

### Scalability Features
- Asynchronous web scraping for improved performance
- Modular pipeline architecture for easy extension
- Structured data models for consistent output format
- Error handling and validation throughout the pipeline

## Key Features

### Medical AI Focus
- Specialized fields for clinical oversight and regulatory approval
- Dermatology-specific bias analysis (Fitzpatrick skin type scale)
- Clinical workflow integration documentation
- Healthcare-specific risk assessment categories

### Bias Detection
- Multi-dimensional bias analysis across demographics
- Source reliability scoring to identify potentially biased information
- Subgroup analysis tracking for fairness assessment
- Clinical risk level categorization for safety assessment

### User Experience
- Multi-page interface for different workflow stages
- Real-time analysis and feedback
- Manual entry options for incomplete automated discovery
- Export capabilities for generated model cards

## Recent Changes: Latest modifications with dates

### 2025-07-16: Model Card Evaluation Feature Added
- **New Feature**: Added comprehensive Model Card Evaluation page for accuracy assessment
- **Input Methods**: Support for text input, JSON upload, and URL/website fetching
- **Automated Discovery**: Uses CardGen pipeline to discover reference information about models
- **Accuracy Assessment**: Compares input model cards against discovered information from multiple sources
- **Detailed Analysis**: Field-by-field comparison with coverage percentages and missing field detection
- **Evaluation Reports**: Downloadable JSON reports with accuracy scores and recommendations
- **Side-by-Side Comparison**: Detailed comparison view for specific fields
- **User Experience**: Progress tracking and comprehensive results display
- **Enhanced Functionality**: Now supports both model card generation AND evaluation workflows

### 2025-07-16: Enhanced Model Card Generation with Detailed Content
- **Comprehensive Enhancement**: Added 11 detailed generation methods for comprehensive field coverage
- **Expanded Uses & Directions**: Detailed intended use, clinical workflow, primary users, how-to-use instructions
- **Enhanced Warnings**: Comprehensive bias considerations, clinical risk assessment, and limitations
- **Technical Specifications**: Detailed model type, data sources, and development data characterization
- **Bias Mitigation**: Extensive bias mitigation approaches and ongoing maintenance descriptions
- **Professional Content**: Clinically relevant, professional-grade content for all major model card sections
- **Improved User Experience**: Much more detailed and comprehensive model card generation

### 2025-07-16: CardGen Pipeline Implementation with CHAI Schema
- **Revolutionary Update**: Implemented fully automated CardGen pipeline method
- **CHAI Schema Integration**: Added complete CHAI Applied Model Card schema v0.1 for HTI-1 and OCR compliance
- **Automated Discovery**: CardGen pipeline searches HuggingFace, GitHub, PubMed, and ArXiv simultaneously
- **Content Extraction**: Intelligent content processing with metrics extraction, clinical info parsing, and reference detection
- **Multi-format Export**: Support for JSON, Markdown, and XML export formats
- **Real-time Progress**: Live progress tracking with step-by-step status updates
- **Streamlined UX**: Single-click model card generation from model name input only
- **Enhanced Architecture**: Added three new core modules:
  - `chai_schema.py`: CHAI-compliant data structures and XML/JSON/Markdown export
  - `cardgen_pipeline.py`: Automated discovery and extraction pipeline
  - Updated `app.py`: New CardGen Pipeline page with real-time progress tracking
- **Compliance Standards**: Full HTI-1 and OCR compliance through CHAI schema implementation
- **User Experience**: Minimal manual input required - only model name confirmation needed

### 2025-07-16: Enhanced Compliance with 45 CFR 170.315
- **Major Update**: Expanded ModelCardData class to include full compliance with 45 CFR 170.315 medical AI requirements
- **Added 60+ new fields** covering all mandatory sections:
  - Metadata with CFR reference tracking
  - Uses & Directions with clinical workflow details
  - Warnings with detailed risk assessment
  - Trust Ingredients with AI system facts
  - Development data with USCDI v3 variables
  - Bias mitigation approaches
  - Ongoing maintenance requirements
  - Key metrics for usefulness, fairness, and safety
  - Transparency information with funding sources
  - Comprehensive resources section
- **Enhanced Manual Entry**: Updated form to include all compliance fields with proper categorization
- **Updated Display Logic**: Restructured model card display to show all sections with CFR references
- **USCDI v3 Integration**: Added specific fields for race, ethnicity, language, sexual orientation, gender identity, social determinants
- **Performance Metrics**: Added detailed performance tracking with AUROC, accuracy, sensitivity, specificity, F1 scores
- **External Validation**: Added fields for external validation processes and measures