import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json

def get_role_requirements():
    """Get role requirements from user input"""
    st.markdown("### Role Requirements")
    
    # Role selection method
    selection_method = st.radio(
        "How would you like to specify the role?",
        ["Select from predefined roles", "Custom role"],
        horizontal=True
    )
    
    if selection_method == "Select from predefined roles":
        # Predefined roles
        roles = {
            "Software Engineer": {
                "role": "Software Engineer",
                "requirements": """
                - Strong programming skills in one or more languages (e.g., Python, Java, JavaScript)
                - Experience with software development lifecycle and best practices
                - Knowledge of data structures, algorithms, and system design
                - Experience with version control systems (e.g., Git)
                - Good problem-solving and analytical skills
                - Ability to work in an agile team environment
                - Strong communication and collaboration skills
                """
            },
            "Data Scientist": {
                "role": "Data Scientist",
                "requirements": """
                - Strong background in statistics and mathematics
                - Proficiency in Python/R and data analysis libraries
                - Experience with machine learning algorithms and frameworks
                - Data visualization and communication skills
                - Knowledge of SQL and database systems
                - Experience with big data technologies
                - Research and analytical mindset
                """
            },
            "Product Manager": {
                "role": "Product Manager",
                "requirements": """
                - Strong understanding of product development lifecycle
                - Experience with agile methodologies
                - Excellent communication and stakeholder management
                - Data-driven decision making skills
                - Market analysis and competitive research abilities
                - Technical background or understanding
                - Leadership and team coordination skills
                """
            },
            "DevOps Engineer": {
                "role": "DevOps Engineer",
                "requirements": """
                - Strong knowledge of CI/CD pipelines
                - Experience with cloud platforms (AWS, Azure, GCP)
                - Container orchestration (Kubernetes, Docker)
                - Infrastructure as Code (Terraform, CloudFormation)
                - Linux/Unix system administration
                - Scripting and automation skills
                - Security best practices knowledge
                """
            }
        }
        
        # Role selection
        selected_role = st.selectbox(
            "Select Role",
            list(roles.keys())
        )
        
        if selected_role:
            # Show requirements
            st.markdown("#### Role Requirements:")
            st.markdown(roles[selected_role]["requirements"])
            return roles[selected_role]
            
    else:
        # Custom role input
        custom_role = st.text_input("Enter Role Title")
        custom_requirements = st.text_area(
            "Enter Role Requirements",
            placeholder="""Example format:
- Required technical skills
- Years of experience needed
- Key responsibilities
- Desired qualifications
- Soft skills required
            """
        )
        
        if custom_role and custom_requirements:
            return {
                "role": custom_role,
                "requirements": custom_requirements
            }
    
    return None

def setup_page():
    """Setup the page with custom CSS"""
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            padding-right: 1rem;
            padding-left: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("AI Resume Analyzer")
    st.markdown("""
    Upload your resume and get detailed analysis, interview questions, and improvement suggestions.
    Each functionality uses a dedicated API key for optimal performance.
    """)

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

def file_uploader_section():
    """Create the file uploader section for resumes"""
    st.markdown("### Upload Resume")
    
    # Instructions
    st.markdown("""
    Upload your resume in PDF, DOCX, or TXT format. 
    The file will be analyzed based on the selected role requirements.
    """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose resume file(s)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple resumes to analyze them together"
    )
    
    if uploaded_files:
        st.success(f"Successfully uploaded {len(uploaded_files)} file(s)")
        
        # Display file details
        for file in uploaded_files:
            with st.expander(f"File: {file.name}"):
                st.write(f"Size: {file.size/1024:.1f} KB")
                st.write(f"Type: {file.type}")
    else:
        st.info("No files uploaded yet")
    
    return uploaded_files

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

def display_analysis_results(analysis_result):
    """Display the resume analysis results"""
    if not analysis_result:
        st.error("No analysis results available")
        return
        
    try:
        # Convert analysis_result to dict if it's a string
        if isinstance(analysis_result, str):
            try:
                analysis_text = analysis_result.strip()
                if analysis_text.startswith("```json"):
                    analysis_text = analysis_text[7:]
                if analysis_text.endswith("```"):
                    analysis_text = analysis_text[:-3]
                analysis_result = json.loads(analysis_text.strip())
            except json.JSONDecodeError:
                st.error("Could not parse analysis results")
                st.text_area("Raw Response", analysis_result, height=300)
                return
                
        # Display match score with progress bar
        match_score = float(analysis_result.get('match_score', 0))
        st.markdown(f"### Overall Match Score: {match_score:.1f}%")
        st.progress(match_score / 100)
        
        # Create three columns for skills, strengths, and weaknesses
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Skills")
            skills = analysis_result.get('extracted_skills', [])
            if isinstance(skills, list):
                for skill in skills:
                    st.markdown(f"- {skill}")
            else:
                st.markdown(f"- {skills}")
                
        with col2:
            st.markdown("#### Strengths")
            strengths = analysis_result.get('strengths', [])
            if isinstance(strengths, list):
                for strength in strengths:
                    st.markdown(f"- {strength}")
            else:
                st.markdown(f"- {strengths}")
                
        with col3:
            st.markdown("#### Weaknesses")
            weaknesses = analysis_result.get('weaknesses', [])
            if isinstance(weaknesses, list):
                for weakness in weaknesses:
                    st.markdown(f"- {weakness}")
            else:
                st.markdown(f"- {weaknesses}")
                
        # Display improvement suggestions
        st.markdown("#### Improvement Suggestions")
        suggestions = analysis_result.get('improvement_suggestions', {})
        if isinstance(suggestions, dict):
            cols = st.columns(len(suggestions))
            for i, (area, items) in enumerate(suggestions.items()):
                with cols[i]:
                    st.markdown(f"**{area.title()}**")
                    if isinstance(items, list):
                        for item in items:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown(f"- {items}")
        else:
            st.markdown("No improvement suggestions available")
            
    except Exception as e:
        st.error(f"Error displaying analysis results: {str(e)}")
        st.text_area("Raw Response", str(analysis_result), height=300)

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
    st.subheader("Generate Interview Questions")
    
    if not has_resume:
        st.warning("Please analyze a resume first.")
        return
        
    # Question types
    question_types = st.multiselect(
        "Select Question Types",
        ["Technical", "Behavioral", "Problem Solving", "Experience", "Project Discussion"],
        default=["Technical", "Behavioral"]
    )
    
    # Difficulty level
    difficulty = st.select_slider(
        "Select Difficulty Level",
        options=["Easy", "Medium", "Hard"],
        value="Medium"
    )
    
    # Number of questions
    num_questions = st.slider(
        "Number of Questions",
        min_value=1,
        max_value=10,
        value=5
    )
    
    if st.button("Generate Questions"):
        with st.spinner("Generating interview questions..."):
            questions = generate_questions_func(question_types, difficulty, num_questions)
            
            if questions:
                st.success("Questions generated successfully!")
                
                # Parse questions from the response
                try:
                    # Clean up the response
                    questions_text = questions.strip()
                    if questions_text.startswith("```json"):
                        questions_text = questions_text[7:]
                    if questions_text.endswith("```"):
                        questions_text = questions_text[:-3]
                    questions_text = questions_text.strip()
                    
                    # Try to parse as JSON first
                    try:
                        questions_data = json.loads(questions_text)
                        if isinstance(questions_data, list):
                            parsed_questions = questions_data
                        else:
                            parsed_questions = questions_data.get('questions', [])
                    except json.JSONDecodeError:
                        # If not JSON, try to parse line by line
                        parsed_questions = []
                        current_type = None
                        current_question = ""
                        
                        for line in questions_text.split('\n'):
                            line = line.strip()
                            if any(t in line for t in question_types):
                                if current_type and current_question:
                                    parsed_questions.append({
                                        "type": current_type,
                                        "question": current_question.strip()
                                    })
                                current_type = next((t for t in question_types if t in line), None)
                                current_question = line.split(":", 1)[1].strip() if ":" in line else ""
                            elif line and current_type:
                                current_question += " " + line
                        
                        if current_type and current_question:
                            parsed_questions.append({
                                "type": current_type,
                                "question": current_question.strip()
                            })
                    
                    # Display questions grouped by type
                    for q_type in question_types:
                        type_questions = [q for q in parsed_questions if q.get('type', '').lower() == q_type.lower()]
                        if type_questions:
                            st.markdown(f"#### {q_type} Questions")
                            for i, q in enumerate(type_questions, 1):
                                st.markdown(f"{i}. {q['question']}")
                            st.markdown("---")
                            
                except Exception as e:
                    st.error(f"Error parsing questions: {str(e)}")
                    st.text_area("Raw Response", questions, height=300)
            else:
                st.error("Failed to generate questions. Please try again.")

def resume_improvement_section(has_resume=False, improve_resume_func=None):
    """Display the resume improvement section"""
    st.markdown("### Resume Improvement Suggestions")
    
    if not has_resume:
        st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")
        return
        
    if not improve_resume_func:
        st.error("Resume improvement function not available.")
        return
        
    if st.button("Get Improvement Suggestions"):
        with st.spinner("Generating improvement suggestions..."):
            improvements = improve_resume_func()
            
            if improvements and 'content' in improvements:
                content = improvements['content']
                
                # Display weakness improvements first
                if 'weaknesses_improvement' in content and content['weaknesses_improvement']:
                    st.markdown("#### 🎯 Detailed Weakness Analysis & Improvements")
                    for item in content['weaknesses_improvement']:
                        with st.expander(f"💡 {item['weakness']}"):
                            st.markdown("**Why this needs improvement:**")
                            st.markdown(item['detailed_explanation'])
                            st.markdown("\n**Steps to Improve:**")
                            for step in item['improvement_steps']:
                                st.markdown(f"- {step}")
                
                # Display other improvements in columns
                st.markdown("#### 📈 Additional Improvement Areas")
                cols = st.columns(4)
                
                # Skills improvements
                with cols[0]:
                    st.markdown("**Skills**")
                    for item in content.get('skills', []):
                        st.markdown(f"- {item}")
                
                # Experience improvements
                with cols[1]:
                    st.markdown("**Experience**")
                    for item in content.get('experience', []):
                        st.markdown(f"- {item}")
                
                # Education improvements
                with cols[2]:
                    st.markdown("**Education**")
                    for item in content.get('education', []):
                        st.markdown(f"- {item}")
                
                # Format improvements
                with cols[3]:
                    st.markdown("**Format**")
                    for item in content.get('format', []):
                        st.markdown(f"- {item}")
            else:
                st.error("Failed to generate improvement suggestions")

def improved_resume_section(has_resume=False, get_improved_resume_func=None):
    """Display the improved resume section"""
    st.markdown("### Improved Resume")
    
    if not has_resume:
        st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")
        return
        
    if not get_improved_resume_func:
        st.error("Resume improvement function not available.")
        return
        
    # Get target role and skills
    target_role = st.text_input("Target Role (optional)")
    highlight_skills = st.text_area("Skills to Highlight (optional, one per line)")
    
    if st.button("Generate Improved Resume"):
        with st.spinner("Generating improved resume..."):
            # Convert skills text to list
            skills_list = [s.strip() for s in highlight_skills.split('\n') if s.strip()] if highlight_skills else None
            
            # Get improved resume
            improved_resume = get_improved_resume_func(target_role, skills_list)
            
            if improved_resume and 'sections' in improved_resume:
                sections = improved_resume['sections']
                
                # Display professional summary
                st.markdown("#### Professional Summary")
                st.write(sections.get('professional_summary', ''))
                
                # Display skills
                st.markdown("#### Skills")
                for skill in sections.get('skills', []):
                    st.markdown(f"- {skill}")
                
                # Display experience
                st.markdown("#### Experience")
                for exp in sections.get('experience', []):
                    st.markdown(f"**{exp.get('title')} at {exp.get('company')}**")
                    st.markdown(f"*{exp.get('duration')}*")
                    for achievement in exp.get('achievements', []):
                        st.markdown(f"- {achievement}")
                
                # Display education
                st.markdown("#### Education")
                for edu in sections.get('education', []):
                    st.markdown(f"**{edu.get('degree')} - {edu.get('institution')}**")
                    st.markdown(f"*{edu.get('duration')}*")
                    for detail in edu.get('details', []):
                        st.markdown(f"- {detail}")
                
                # Display certifications
                if sections.get('certifications'):
                    st.markdown("#### Certifications")
                    for cert in sections.get('certifications', []):
                        st.markdown(f"- {cert.get('name')} from {cert.get('issuer')} ({cert.get('date')})")
                
                # Create downloadable version
                resume_text = f"""# Professional Resume

## Professional Summary
{sections.get('professional_summary', '')}

## Skills
{chr(10).join('- ' + skill for skill in sections.get('skills', []))}

## Experience
{chr(10).join(f'''### {exp.get('title')} at {exp.get('company')}
{exp.get('duration')}
{chr(10).join("- " + achievement for achievement in exp.get('achievements', []))}''' for exp in sections.get('experience', []))}

## Education
{chr(10).join(f'''### {edu.get('degree')} - {edu.get('institution')}
{edu.get('duration')}
{chr(10).join("- " + detail for detail in edu.get('details', []))}''' for edu in sections.get('education', []))}

## Certifications
{chr(10).join(f"- {cert.get('name')} from {cert.get('issuer')} ({cert.get('date')})" for cert in sections.get('certifications', []))}
"""
                
                # Download button for the resume
                st.download_button(
                    label="Download Improved Resume",
                    data=resume_text,
                    file_name="improved_resume.md",
                    mime="text/markdown"
                )
            else:
                st.error("Failed to generate improved resume")
