import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def setup_page():
    """Setup the page with custom CSS"""
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #4CAF50;
            color: black;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
        }
        .stSelectbox>div>div>select {
            border-radius: 5px;
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        .stProgress .st-bo {
            background-color: #4CAF50;
        }
        .skill-score {
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
            background-color: #f0f2f6;
        }
        .keyword-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
        .keyword {
            display: inline-block;
            background-color: #e1e4e8;
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            border-radius: 15px;
            font-size: 0.9em;
        }
        </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the application header"""
    st.title("🤖 AI Resume Analysis & Interview Preparation")
    st.markdown("""
    Upload your resume and get instant AI-powered analysis, interview questions, and improvement suggestions.
    """)

def setup_sidebar():
    """Setup the sidebar with configuration options"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")
        st.markdown("""
        ### About
        This tool uses AI to:
        - Analyze resumes
        - Generate interview questions
        - Provide improvement suggestions
        - Create optimized versions
        """)
        
    return {"openai_api_key": None}  # Return None since we'll use hardcoded key

def create_tabs():
    """Create the main application tabs"""
    return st.tabs([
        "📄 Resume Analysis",
        "❓ Resume Q&A",
        "🎯 Interview Questions",
        "📈 Resume Improvement",
        "✨ Improved Resume"
    ])

def role_selection_section(role_requirements):
    """Create the role selection section"""
    st.subheader("1. Select Role or Upload Custom Job Description")
    
    # Add checkbox for custom JD upload
    use_custom_jd = st.checkbox("Upload Custom Job Description", value=False)
    
    if use_custom_jd:
        # Create two columns for file upload and manual input
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Option 1: Upload Job Description File")
            custom_jd_file = st.file_uploader(
                "Upload Job Description (PDF or TXT)",
                type=["txt", "pdf"]
            )
            
        with col2:
            st.markdown("### Option 2: Enter Job Description Manually")
            manual_jd = st.text_area(
                "Enter job description or keywords",
                height=200,
                placeholder="Enter the job description, required skills, or keywords here..."
            )
            
        # If both are empty, show warning
        if not custom_jd_file and not manual_jd:
            st.warning("Please either upload a job description file or enter the job description manually.")
            
        role = None
    else:
        # If checkbox is unchecked, show role selection
        role = st.selectbox(
            "Select Target Role",
            options=list(role_requirements.keys()),
            index=0
        )
        custom_jd_file = None
        manual_jd = None
        
        # Display ATS keywords for selected role
        if role:
            st.markdown("### 🔍 ATS Keywords for this Role")
            keywords = role_requirements[role]
            # Create a nice display of keywords
            st.markdown("""
                <style>
                .keyword-box {
                    background-color: #f0f2f6;
                    padding: 1rem;
                    border-radius: 5px;
                    margin: 0.5rem 0;
                }
                .keyword {
                    display: inline-block;
                    background-color: #e1e4e8;
                    padding: 0.3rem 0.8rem;
                    margin: 0.2rem;
                    border-radius: 15px;
                    font-size: 0.9em;
                    color: black;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Display keywords in a nice format
            keywords_html = '<div class="keyword-box">'
            for keyword in keywords:
                keywords_html += f'<span class="keyword">{keyword}</span>'
            keywords_html += '</div>'
            st.markdown(keywords_html, unsafe_allow_html=True)
            
            st.markdown("""
                <div style='font-size: 0.9em; color: black; margin-top: 0.5rem;'>
                    💡 These keywords are important for ATS scanning. Make sure your resume includes relevant experience with these technologies.
                </div>
            """, unsafe_allow_html=True)
    
    return role, custom_jd_file, manual_jd

def resume_upload_section():
    """Create the resume upload section"""
    st.subheader("2. Upload Your Resume(s)")
    
    # Always show the multiple resume info text
    st.markdown("""
        <div style='font-size: 0.9em; color: white; margin-bottom: 0.5rem; background-color: #4CAF50; padding: 0.5rem; border-radius: 5px;'>
            💡 You can upload up to 5 resumes at a time for batch analysis. 
            Each resume will be analyzed separately and results will be displayed in expandable sections.
        </div>
    """, unsafe_allow_html=True)
    
    # Add checkbox for multiple resume upload
    multiple_resumes = st.checkbox("Upload Multiple Resumes", value=False)
    
    if multiple_resumes:
        uploaded_files = st.file_uploader(
            "Upload multiple resumes (PDF or TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        
        # Check if number of files exceeds limit
        if uploaded_files and len(uploaded_files) > 5:
            st.warning("⚠️ Maximum 5 resumes allowed. Only the first 5 will be analyzed.")
            uploaded_files = uploaded_files[:5]
            
        return uploaded_files
    else:
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF or TXT)",
            type=["pdf", "txt"]
        )
        return [uploaded_file] if uploaded_file else []

def display_analysis_results(result):
    """Display the analysis results"""
    st.subheader("Analysis Results")
    
    # Display overall score
    score = result.get('score', 0)
    normalized_score = min(score/100, 1.0)  # Normalize score to be between 0 and 1
    st.progress(normalized_score)
    st.write(f"Overall Match Score: {score:.1f}%")
    
    # Display strengths
    st.subheader("Strengths")
    for strength in result.get('strengths', []):
        st.write(f"✅ {strength}")
    
    # Display weaknesses
    st.subheader("Areas for Improvement")
    for weakness in result.get('weaknesses', []):
        st.write(f"⚠️ {weakness}")
    
    # Display improvement suggestions
    st.subheader("Improvement Suggestions")
    for category, suggestions in result.get('improvement_suggestions', {}).items():
        st.write(f"**{category}:**")
        for suggestion in suggestions:
            st.write(f"- {suggestion}")
    
    # Display ATS keywords
    st.subheader("ATS Keywords")
    for keyword in result.get('ats_keywords', []):
        st.write(f"🔑 {keyword}")

def resume_qa_section(has_resume, ask_question_func):
    """Create the resume Q&A section"""
    if not has_resume:
        st.warning("Please upload and analyze a resume first.")
        return
        
    st.markdown("### Ask Questions About the Resume")
    question = st.text_input("Enter your question:")
    
    if st.button("Ask Question"):
        if question:
            response = ask_question_func(question)
            st.markdown("#### Answer:")
            st.markdown(response)
        else:
            st.error("Please enter a question.")

def interview_questions_section(has_resume, generate_questions_func):
    """Create the interview questions section"""
    if not has_resume:
        st.warning("Please upload and analyze a resume first.")
        return
        
    st.markdown("### Generate Interview Questions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        question_types = st.multiselect(
            "Question Types",
            ["Technical", "Behavioral", "Problem Solving", "Experience", "Project"],
            default=["Technical", "Behavioral"]
        )
        
    with col2:
        difficulty = st.select_slider(
            "Difficulty Level",
            options=["Easy", "Medium", "Hard"],
            value="Medium"
        )
        
    with col3:
        num_questions = st.number_input(
            "Number of Questions",
            min_value=1,
            max_value=10,
            value=5
        )
        
    if st.button("Generate Questions"):
        if question_types:
            questions = generate_questions_func(question_types, difficulty, num_questions)
            for q_type, question in questions:
                with st.expander(f"{q_type} Question"):
                    st.markdown(question)
        else:
            st.error("Please select at least one question type.")

def resume_improvement_section(has_resume, improve_resume_func):
    """Create the resume improvement section"""
    if not has_resume:
        st.warning("Please upload and analyze a resume first.")
        return
        
    st.markdown("### Get Resume Improvement Suggestions")
    
    improvement_areas = st.multiselect(
        "Select areas for improvement",
        [
            "Skills Highlighting",
            "Experience Description",
            "Achievement Quantification",
            "Format and Structure",
            "Professional Summary"
        ],
        default=["Skills Highlighting", "Achievement Quantification"]
    )
    
    target_role = st.text_input("Target Role (optional):")
    
    if st.button("Get Improvement Suggestions"):
        if improvement_areas:
            improvements = improve_resume_func(improvement_areas, target_role)
            
            for area, details in improvements.items():
                with st.expander(f"📝 {area}"):
                    st.markdown(f"**Overview:** {details['description']}")
                    
                    st.markdown("\n**Specific Suggestions:**")
                    for suggestion in details.get('specific', []):
                        st.markdown(f"- {suggestion}")
                        
                    if 'before_after' in details and details['before_after']:
                        st.markdown("\n**Example Improvement:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Before:**")
                            st.markdown(details['before_after']['before'])
                        with col2:
                            st.markdown("**After:**")
                            st.markdown(details['before_after']['after'])
        else:
            st.error("Please select at least one improvement area.")

def improved_resume_section(has_resume, get_improved_resume_func):
    """Create the improved resume section"""
    if not has_resume:
        st.warning("Please upload and analyze a resume first.")
        return
        
    st.markdown("### Get AI-Improved Resume")
    
    target_role = st.text_input("Target Role:")
    highlight_skills = st.text_area(
        "Skills to Highlight (comma-separated) or paste full job description:",
        height=100
    )
    
    if st.button("Generate Improved Resume"):
        improved_resume = get_improved_resume_func(target_role, highlight_skills)
        st.markdown("### Improved Resume")
        st.markdown("---")
        st.markdown(improved_resume)
        
        # Download button for the improved resume
        st.download_button(
            label="Download Improved Resume",
            data=improved_resume,
            file_name="improved_resume.txt",
            mime="text/plain"
        )
