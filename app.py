import streamlit as st
# This MUST be the first Streamlit command
st.set_page_config(
    page_title="Euron Recruitment Agent",
    page_icon="",
    layout="wide"
)

import os
from dotenv import load_dotenv
import ui
from agents import ResumeAnalysisAgent
import atexit
import tempfile

# Load environment variables from .env file
load_dotenv(override=True)  # Add override=True to ensure values are updated

# Check for all required API keys
API_KEYS = {
    'API_KEY_ANALYSIS': os.getenv('API_KEY_ANALYSIS'),
    'API_KEY_QA': os.getenv('API_KEY_QA'),
    'API_KEY_QUESTIONS': os.getenv('API_KEY_QUESTIONS'),
    'API_KEY_IMPROVEMENT': os.getenv('API_KEY_IMPROVEMENT'),
    'API_KEY_IMPROVED_RESUME': os.getenv('API_KEY_IMPROVED_RESUME')
}

# Debug: Print loaded keys
print("Loaded API Keys:")
for key_name, key_value in API_KEYS.items():
    if key_value:
        print(f"[+] {key_name}: {key_value[:10]}...")
    else:
        print(f"[-] {key_name}: Not found")

# Verify at least one API key is available
available_keys = [k for k, v in API_KEYS.items() if v]
if not available_keys:
    st.error("No API keys found in environment. Please set at least one API key in the .env file.")
    st.stop()
else:
    print(f"Successfully loaded {len(available_keys)} API keys")

# Initialize session state variables
if 'resume_agent' not in st.session_state:
    try:
        st.session_state.resume_agent = ResumeAnalysisAgent()
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        st.stop()
if 'resume_analyzed' not in st.session_state:
    st.session_state.resume_analyzed = False
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

def setup_agent(config):
    """Set up the resume analysis agent"""
    if st.session_state.resume_agent is None:
        st.session_state.resume_agent = ResumeAnalysisAgent()
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

def main():
    # Setup the page
    ui.setup_page()
    
    # Initialize session state
    if 'resume_agent' not in st.session_state:
        try:
            st.session_state.resume_agent = ResumeAnalysisAgent()
        except Exception as e:
            st.error(f"Error initializing agent: {str(e)}")
            st.stop()
    if 'resume_analyzed' not in st.session_state:
        st.session_state.resume_analyzed = False
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
        
    # Create tabs
    tabs = st.tabs([
        "Resume Analysis",
        "Resume Q&A",
        "Interview Questions",
        "Resume Improvement",
        "Improved Resume"
    ])
    
    # Resume Analysis Tab
    with tabs[0]:
        st.title("Resume Analysis")
        
        # Get role requirements
        role_requirements = ui.get_role_requirements()
        
        # File uploader for resumes
        uploaded_files = ui.file_uploader_section()
        
        if uploaded_files and role_requirements:
            if st.button("Analyze Resume(s)"):
                with st.spinner("Analyzing resume(s)..."):
                    all_results = []
                    
                    for uploaded_file in uploaded_files:
                        # Analyze the resume
                        result = st.session_state.resume_agent.analyze_resume(
                            uploaded_file,
                            role_requirements
                        )
                        
                        if result:
                            all_results.append({
                                'filename': uploaded_file.name,
                                'result': result
                            })
                        
                    # Set analysis flag and results if successful
                    if all_results:
                        st.session_state.resume_analyzed = True
                        st.session_state.analysis_result = all_results[0]['result']  # Store first result
                        
                        # Display results for each resume
                        for resume_data in all_results:
                            with st.expander(f"Analysis Results for {resume_data['filename']}"):
                                ui.display_analysis_results(resume_data['result'])
                    else:
                        st.error("Failed to analyze resume(s)")
                        
    # Resume Q&A Tab
    with tabs[1]:
        ui.resume_qa_section(
            has_resume=st.session_state.resume_analyzed,
            ask_question_func=lambda q: st.session_state.resume_agent.ask_question(q)
        )
        
    # Interview Questions Tab
    with tabs[2]:
        ui.interview_questions_section(
            has_resume=st.session_state.resume_analyzed,
            generate_questions_func=lambda types, diff, num: st.session_state.resume_agent.generate_interview_questions(types, diff, num)
        )
        
    # Resume Improvement Tab
    with tabs[3]:
        ui.resume_improvement_section(
            has_resume=st.session_state.resume_analyzed,
            improve_resume_func=lambda areas="", role="": st.session_state.resume_agent.improve_resume(areas, role)
        )
        
    # Improved Resume Tab
    with tabs[4]:
        ui.improved_resume_section(
            has_resume=st.session_state.resume_analyzed,
            get_improved_resume_func=lambda role, skills: st.session_state.resume_agent.get_improved_resume(role, skills)
        )

if __name__ == "__main__":
    main()