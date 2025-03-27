import streamlit as st
# This MUST be the first Streamlit command
st.set_page_config(
    page_title="Euron Recruitment Agent",
    page_icon="🤖",
    layout="wide"
)

import os
from dotenv import load_dotenv
import ui
from agents import ResumeAnalysisAgent
import atexit
import tempfile

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment. Please set it in the .env file.")
    st.stop()

# Role requirements dictionary
ROLE_REQUIREMENTS = {
    "AI/ML Engineer": [
        "Python", "PyTorch", "TensorFlow", "Machine Learning", "Deep Learning",
        "MLOps", "Scikit-Learn", "NLP", "Computer Vision", "Reinforcement Learning",
        "Hugging Face", "Data Engineering", "Feature Engineering", "AutoML"
    ],
    "Frontend Engineer": [
        "React", "Vue", "Angular", "HTML5", "CSS3", "JavaScript", "TypeScript",
        "Next.js", "Svelte", "Bootstrap", "Tailwind CSS", "GraphQL", "Redux",
        "WebAssembly", "Three.js", "Performance Optimization"
    ],
    "Backend Engineer": [
        "Node.js", "Express", "Django", "Flask", "RESTful APIs", "Database Design",
        "SQL", "NoSQL", "Microservices", "Docker", "Kubernetes", "AWS", "CI/CD",
        "Git", "GitHub", "GitLab", "Docker"
    ],
    "Data Engineer": [
        "Python", "SQL", "Apache Spark", "Hadoop", "Kafka", "ETL Pipelines",
        "Airflow", "BigQuery", "Redshift", "Data Warehousing", "Snowflake",
        "Azure Data Factory", "GCP", "AWS Glue", "DBT"
    ],
    "DevOps Engineer": [
        "Kubernetes", "Docker", "Terraform", "CI/CD", "AWS", "Azure", "GCP",
        "Jenkins", "Ansible", "Prometheus", "Grafana", "Helm", "Linux Administration",
        "Networking", "Site Reliability Engineering (SRE)"
    ],
    "Full Stack Engineer": [
        "React", "Vue", "Angular", "Node.js", "Express", "Django", "Flask",
        "RESTful APIs", "Database Design", "SQL", "NoSQL", "Microservices",
        "Docker", "Kubernetes", "AWS", "CI/CD", "Git", "GitHub", "GitLab"
    ],
    "Cloud Engineer": [
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "CI/CD",
        "Networking", "Linux Administration", "Cloud Security", "Cloud Architecture"
    ],
    "Data Scientist": [
        "Python", "SQL", "Apache Spark", "Hadoop", "Kafka", "ETL Pipelines",
        "Airflow", "BigQuery", "Redshift", "Data Warehousing", "Snowflake",
        "Azure Data Factory", "GCP", "AWS Glue", "DBT"
    ],
    "Cybersecurity Engineer": [
        "Cybersecurity", "Network Security", "Cloud Security", "DevSecOps",
        "Firewalls", "IDS/IPS", "VPNs", "SIEM", "SOAR", "Threat Detection",
        "Incident Response", "Cloud Security"
    ],
    "Product Manager": [
        "Product Management", "Agile Methodologies", "User Stories", "Scrum",
        "Kanban", "Jira", "Confluence", "User Research", "Market Analysis",
        "Product Strategy", "User Testing", "A/B Testing"
    ]
}

# Initialize session state variables
if 'resume_agent' not in st.session_state:
    st.session_state.resume_agent = None
if 'resume_analyzed' not in st.session_state:
    st.session_state.resume_analyzed = False
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

def setup_agent(config):
    """Set up the resume analysis agent with the provided configuration"""
    if st.session_state.resume_agent is None:
        st.session_state.resume_agent = ResumeAnalysisAgent(api_key=GROQ_API_KEY)
    else:
        st.session_state.resume_agent.api_key = GROQ_API_KEY
    return st.session_state.resume_agent

def analyze_resume(agent, resume_file, role, custom_jd=None, manual_jd=None):
    """Analyze the resume with the agent"""
    if not resume_file:
        st.error("Please upload a resume.")
        return None
    
    try:
        with st.spinner("Analyzing resume... This may take a minute."):
            if custom_jd:
                result = agent.analyze_resume(resume_file, custom_jd=custom_jd)
            elif manual_jd:
                # Create a temporary file with the manual JD
                with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as tmp:
                    tmp.write(manual_jd)
                    tmp_jd_path = tmp.name
                
                # Use the temporary file for analysis
                result = agent.analyze_resume(resume_file, custom_jd=tmp_jd_path)
                
                # Clean up the temporary file
                os.unlink(tmp_jd_path)
            else:
                # Create a role requirements dictionary
                role_reqs = {
                    "role": role,
                    "requirements": "\n".join(ROLE_REQUIREMENTS[role])
                }
                result = agent.analyze_resume(resume_file, custom_jd=role_reqs)
            return result
    except Exception as e:
        st.error(f"Error analyzing resume: {e}")
        return None

def ask_question(agent, question):
    """Ask a question about the resume"""
    try:
        with st.spinner("Generating response..."):
            response = agent.ask_question(question)
            return response
    except Exception as e:
        return f"Error: {e}"

def generate_interview_questions(agent, question_types, difficulty, num_questions):
    """Generate interview questions based on the resume"""
    try:
        with st.spinner("Generating personalized interview questions..."):
            questions = agent.generate_interview_questions(question_types, difficulty, num_questions)
            return questions
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []

def improve_resume(agent, improvement_areas, target_role):
    """Generate resume improvement suggestions"""
    try:
        with st.spinner("Analyzing and generating improvements..."):
            return agent.improve_resume(improvement_areas, target_role)
    except Exception as e:
        st.error(f"Error generating improvements: {e}")
        return {}

def get_improved_resume(agent, target_role, highlight_skills):
    """Get an improved version of the resume"""
    try:
        with st.spinner("Creating improved resume..."):
            return agent.get_improved_resume(target_role, highlight_skills)
    except Exception as e:
        st.error(f"Error creating improved resume: {e}")
        return "Error generating improved resume."

def cleanup():
    """Clean up resources when the app exits"""
    if st.session_state.resume_agent:
        st.session_state.resume_agent.cleanup()

# Register cleanup function
atexit.register(cleanup)

def main():
    # Setup page UI
    ui.setup_page()
    ui.display_header()
    
    # Set up sidebar and get configuration
    config = ui.setup_sidebar()
    
    # Set up the agent
    agent = setup_agent(config)
    
    # Create tabs for different functionalities
    tabs = ui.create_tabs()

    # Tab 1: Resume Analysis
    with tabs[0]:
        role, custom_jd, manual_jd = ui.role_selection_section(ROLE_REQUIREMENTS)
        uploaded_resumes = ui.resume_upload_section()
        
        if uploaded_resumes:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Analyze Resume(s)", type="primary"):
                    if agent:
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Store results for each resume
                        all_results = []
                        
                        # Analyze each resume
                        for i, resume in enumerate(uploaded_resumes):
                            status_text.text(f"Analyzing resume {i+1} of {len(uploaded_resumes)}...")
                            result = analyze_resume(agent, resume, role, custom_jd, manual_jd)
                            if result:
                                all_results.append({
                                    "filename": resume.name,
                                    "result": result
                                })
                            progress_bar.progress((i + 1) / len(uploaded_resumes))
                        
                        # Display results for each resume
                        for resume_data in all_results:
                            with st.expander(f"📄 Analysis Results for {resume_data['filename']}"):
                                ui.display_analysis_results(resume_data['result'])
                        
                        # Set resume_analyzed flag after successful analysis
                        if all_results:
                            st.session_state.resume_analyzed = True
                        status_text.text("Analysis complete!")
                    else:
                        st.error("Error initializing the resume analysis agent. Please try again.")

    # Tab 2: Resume Q&A
    with tabs[1]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            st.info("Using the last analyzed resume for Q&A.")
            ui.resume_qa_section(
                has_resume=True,
                ask_question_func=lambda q: ask_question(st.session_state.resume_agent, q)
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

    # Tab 3: Interview Questions
    with tabs[2]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            st.info("Using the last analyzed resume for generating interview questions.")
            ui.interview_questions_section(
                has_resume=True,
                generate_questions_func=lambda types, diff, num: generate_interview_questions(
                    st.session_state.resume_agent, types, diff, num
                )
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

    # Tab 4: Resume Improvement
    with tabs[3]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            st.info("Using the last analyzed resume for improvement suggestions.")
            ui.resume_improvement_section(
                has_resume=True,
                improve_resume_func=lambda areas, role: improve_resume(
                    st.session_state.resume_agent, areas, role
                )
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

    # Tab 5: Improved Resume
    with tabs[4]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            st.info("Using the last analyzed resume for generating an improved version.")
            ui.improved_resume_section(
                has_resume=True,
                get_improved_resume_func=lambda role, skills: get_improved_resume(
                    st.session_state.resume_agent, role, skills
                )
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

if __name__ == "__main__":
    main()