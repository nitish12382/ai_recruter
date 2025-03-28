import re
import PyPDF2
import io
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional

# Load environment variables from .env file
load_dotenv()

# Custom Groq LLM class for LangChain compatibility
class GroqLLM(LLM):
    client: Groq
    model: str = "llama-3.3-70b-specdec"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        return completion.choices[0].message.content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "groq"

class ResumeAnalysisAgent:
    def __init__(self):
        """Initialize the ResumeAnalysisAgent with multiple API keys"""
        # Load environment variables
        load_dotenv()  # Ensure environment variables are loaded
        
        self.api_keys = {
            'analysis': os.getenv('API_KEY_ANALYSIS'),
            'qa': os.getenv('API_KEY_QA'),
            'questions': os.getenv('API_KEY_QUESTIONS'),
            'improvement': os.getenv('API_KEY_IMPROVEMENT'),
            'improved_resume': os.getenv('API_KEY_IMPROVED_RESUME')
        }
        
        # Initialize clients for each functionality
        self.clients = {}
        for key_type, api_key in self.api_keys.items():
            if api_key:
                self.clients[key_type] = Groq(api_key=api_key)
                print(f"Initialized client for {key_type} with key {api_key[:10]}...")  # Debug print
        
        if not self.clients:
            raise ValueError("No valid API keys found. Please check your .env file.")
            
        self.vector_store = None
        self.resume_text = None
        self.role_requirements = None
        self.analysis_cache = {}
        self.current_key_index = 0
        self.extracted_skills = None
        self.analysis_result = None
        self.cutoff_score = 70  # Define the cutoff score
        
    def _get_client(self, functionality):
        """Get client for specific functionality with fallback and rotation"""
        if not self.clients:
            raise ValueError("No API keys configured")
            
        # Try primary client for the functionality
        if functionality in self.clients:
            return self.clients[functionality]
            
        # Fallback to any available client using rotation
        available_clients = list(self.clients.values())
        if not available_clients:
            raise ValueError("No available API clients")
            
        self.current_key_index = (self.current_key_index + 1) % len(available_clients)
        return available_clients[self.current_key_index]
    
    def _get_llm(self, functionality):
        """Get LLM instance for specific functionality"""
        client = self._get_client(functionality)
        return GroqLLM(client=client)
    
    def _handle_api_call(self, functionality, operation):
        """Handle API calls with retries and error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = self._get_client(functionality)
                return operation(client)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
                continue

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from a PDF file"""
        try:
            if hasattr(pdf_file, 'getvalue'):
                pdf_data = pdf_file.getvalue()
                pdf_file_like = io.BytesIO(pdf_data)
                reader = PyPDF2.PdfReader(pdf_file_like)
            else:
                reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def extract_text_from_txt(self, txt_file):
        """Extract text from a text file"""
        try:
            if hasattr(txt_file, 'getvalue'):
                return txt_file.getvalue().decode('utf-8')
            with open(txt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error extracting text from text file: {e}")
            return ""

    def extract_text_from_file(self, file):
        """Extract text from a file (PDF or TXT)"""
        if hasattr(file, 'name'):
            file_extension = file.name.split('.')[-1].lower()
        else:
            file_extension = file.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(file)
        elif file_extension == 'txt':
            return self.extract_text_from_txt(file)
        else:
            print(f"Unsupported file extension: {file_extension}")
            return ""

    def create_rag_vector_store(self, text):
        """Create a vector store for RAG using HuggingFace embeddings"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        vectorstore = FAISS.from_texts(chunks, embedding_model)
        return vectorstore

    def create_vector_store(self, text):
        """Create a simpler vector store for skill analysis"""
        embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        vectorstore = FAISS.from_texts([text], embedding_model)
        return vectorstore

    def analyze_skill(self, qa_chain, skill):
        """Analyze a skill in the resume using the QA chain"""
        prompt = f"Analyze the skill: {skill} in the context of the resume."
        response = qa_chain.run(prompt)
        return response

    def analyze_resume_weaknesses(self):
        """Analyze specific weaknesses in the resume based on missing skills"""
        if not self.resume_text or not self.extracted_skills or not self.analysis_result:
            return []
        
        weaknesses = []
        for skill in self.analysis_result.get("missing_skills", []):
            prompt = f"""
            Analyze why the resume is weak in demonstrating proficiency in {skill}.

            For your analysis, consider:
            1. What's missing from the resume regarding this skill?
            2. How could it be improved with specific examples?
            3. What specific action items would make this skill stand out?
            4. What are the key areas of improvement for this skill?
            5. Grammar and punctuation

            Resume Content:
            {self.resume_text[:3000]} ...

            Provide your response in this JSON format:
            {{
                "weakness": "A concise description of what's missing or problematic (1-2 sentences)",
                "improvement_suggestions": [
                    "Specific suggestion 1",
                    "Specific suggestion 2",
                    "Specific suggestion 3",
                    "Specific suggestion 4",
                    "Specific suggestion 5"
                ],
                "example_addition": "A specific bullet point that could be added to showcase the skill"
            }}

            Return only valid JSON, no other text
            """

            response = self._get_llm('analysis')(prompt)
            weakness_content = response.strip()
            
            try:
                weakness_data = json.loads(weakness_content)
                weakness_detail = {
                    "skill": skill,
                    "score": self.analysis_result.get("skill_scores", {}).get(skill, 0),
                    "detail": weakness_data.get("weakness", "No specific details provided."),
                    "suggestions": weakness_data.get("improvement_suggestions", []),
                    "example": weakness_data.get("example_addition", "")
                }
                weaknesses.append(weakness_detail)
            except json.JSONDecodeError:
                weaknesses.append({
                    "skill": skill,
                    "score": self.analysis_result.get("skill_scores", {}).get(skill, 0),
                    "detail": weakness_content[:200]  # Truncate if it's not proper JSON
                })

        self.resume_weaknesses = weaknesses
        return weaknesses

    def extract_skills_from_jd(self, jd_text):
        """Extract skills from a job description"""
        try:
            prompt = f"""
            Extract a comprehensive list of technical skills, technologies, and competencies required from this job description.
            Format the output as a Python list of strings. Only include the list, nothing else.

            Job Description:
            {jd_text}
            """

            response = self._get_llm('analysis')(prompt)
            skills_text = response
            match = re.search(r'\[(.*?)\]', skills_text, re.DOTALL)
            
            if match:
                skills_text = match.group(0)
                try:
                    skills_list = eval(skills_text)
                    if isinstance(skills_list, list):
                        return skills_list
                except:
                    pass

            skills = []
            for line in skills_text.split('\n'):
                line = line.strip()
                if line.startswith('- ') or line.startswith('* '):
                    skill = line[2:].strip()
                    if skill:
                        skills.append(skill)
                elif line.startswith('"') and line.endswith('"'):
                    skill = line.strip('"')
                    if skill:
                        skills.append(skill)

            return skills
        except Exception as e:
            print(f"Error extracting skills from job description: {e}")
            return []

    def semantic_skill_analysis(self, resume_text, skills):
        """Analyze skills semantically"""
        vectorstore = self.create_vector_store(resume_text)
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=self._get_llm('analysis'),
            retriever=retriever,
            return_source_documents=False
        )
        
        skill_scores = {}
        skill_reasoning = {}
        missing_skills = []
        total_score = 0

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(lambda skill: self.analyze_skill(qa_chain, skill), skills))
            
            for response, skill in zip(results, skills):
                match = re.search(r"(\d{1,2})", response)
                score = int(match.group(1)) if match else 0
                reasoning = response.split('. ', 1)[1].strip() if '.' in response and len(response.split('.')) > 1 else ""
                skill_scores[skill] = score
                skill_reasoning[skill] = reasoning
                total_score += score
                if score <= 5:
                    missing_skills.append(skill)

        overall_score = int((total_score / (10 * len(skills))) * 100)
        selected = overall_score >= self.cutoff_score
        reasoning = "Candidate evaluated based on explicit resume content using semantic similarity and clear numeric scoring."
        strengths = [skill for skill, score in skill_scores.items() if score >= 7]
        improvement_areas = missing_skills if not selected else []

        self.resume_strengths = strengths

        return {
            "overall_score": overall_score,
            "skill_scores": skill_scores,
            "skill_reasoning": skill_reasoning,
            "selected": selected,
            "reasoning": reasoning,
            "missing_skills": missing_skills,
            "strengths": strengths,
            "improvement_areas": improvement_areas
        }

    def analyze_resume(self, resume_file, role_requirements=None, custom_jd=None):
        """Analyze resume with dedicated API key and cache results"""
        try:
            # Extract text and prepare analysis
            self.resume_text = self.extract_text_from_file(resume_file)
            self.role_requirements = role_requirements
            
            # Use analysis-specific API key
            llm = self._get_llm('analysis')
            
            # Analyze resume
            analysis_prompt = f"""Analyze this resume for the role of {self.role_requirements['role'] if self.role_requirements else 'General Role'}:

Resume:
{self.resume_text}

Role Requirements:
{self.role_requirements['requirements'] if self.role_requirements else 'General requirements for resume analysis'}

Provide a detailed analysis in the following JSON format:
{{
    "extracted_skills": [<list of technical and soft skills found in resume>],
    "strengths": [<list of strengths>],
    "weaknesses": [<list of weaknesses>],
    "match_score": <number between 0 and 100, calculate based on role requirements match>,
    "improvement_suggestions": {{
        "technical": [<list of suggestions>],
        "experience": [<list of suggestions>],
        "education": [<list of suggestions>]
    }}
}}

IMPORTANT: Calculate match_score carefully by comparing resume skills and experience with role requirements. If no role requirements provided, base score on general resume quality.
IMPORTANT: Respond ONLY with the JSON object, no additional text or explanations."""
            
            # Get analysis from LLM
            response = self._handle_api_call('analysis', lambda client: 
                client.chat.completions.create(
                    model="llama-3.3-70b-specdec",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.7,
                    max_tokens=1024
                ).choices[0].message.content
            )
            
            # Clean up the response to ensure it's valid JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Parse the response as JSON
            try:
                analysis_result = json.loads(response)
                # Ensure match_score is a number between 0 and 100
                if 'match_score' in analysis_result:
                    try:
                        match_score = float(analysis_result['match_score'])
                        analysis_result['match_score'] = max(0, min(100, match_score))
                    except (ValueError, TypeError):
                        analysis_result['match_score'] = 0
                else:
                    analysis_result['match_score'] = 0
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract the JSON part
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        analysis_result = json.loads(json_match.group())
                        if 'match_score' in analysis_result:
                            try:
                                match_score = float(analysis_result['match_score'])
                                analysis_result['match_score'] = max(0, min(100, match_score))
                            except (ValueError, TypeError):
                                analysis_result['match_score'] = 0
                        else:
                            analysis_result['match_score'] = 0
                    except json.JSONDecodeError:
                        analysis_result = {
                            "extracted_skills": [],
                            "strengths": ["Unable to analyze strengths"],
                            "weaknesses": ["Unable to analyze weaknesses"],
                            "match_score": 0,
                            "improvement_suggestions": {
                                "technical": ["Unable to generate technical suggestions"],
                                "experience": ["Unable to generate experience suggestions"],
                                "education": ["Unable to generate education suggestions"]
                            }
                        }
                else:
                    raise ValueError("Could not parse analysis result as JSON")
            
            # Cache the results for other functionalities
            self.analysis_cache = analysis_result
            self.extracted_skills = analysis_result.get('extracted_skills', [])
            self.analysis_result = analysis_result
            return analysis_result
            
        except Exception as e:
            st.error(f"Error in resume analysis: {str(e)}")
            return None

    def ask_question(self, question):
        """Ask questions about resume using QA-specific API key"""
        try:
            if not self.resume_text:
                raise ValueError("No resume loaded")
                
            # Include cached analysis in prompt context
            context = f"""
            Resume Analysis Context:
            - Skills: {self.analysis_cache.get('extracted_skills', [])}
            - Strengths: {self.analysis_cache.get('strengths', [])}
            - Weaknesses: {self.analysis_cache.get('weaknesses', [])}
            
            Question: {question}
            """
            
            return self._handle_api_call('qa', lambda client: 
                client.chat.completions.create(
                    model="llama-3.3-70b-specdec",
                    messages=[{"role": "user", "content": context}],
                    temperature=0.7,
                    max_tokens=1024
                ).choices[0].message.content
            )
            
        except Exception as e:
            st.error(f"Error in resume Q&A: {str(e)}")
            return None

    def generate_interview_questions(self, question_types, difficulty, num_questions):
        """Generate interview questions based on resume analysis"""
        try:
            llm = self._get_llm('questions')
            
            # Create prompt for interview questions
            prompt = f"""Based on the following resume and role requirements, generate {num_questions} {difficulty}-level interview questions.
            Focus on these types: {', '.join(question_types)}

            Resume:
            {self.resume_text}

            Role Requirements:
            {self.role_requirements['requirements'] if self.role_requirements else 'General requirements'}

            Previous Analysis:
            {json.dumps(self.analysis_cache, indent=2)}

            Generate questions in this JSON format:
            {{
                "questions": [
                    {{
                        "type": "question type",
                        "question": "actual question"
                    }}
                ]
            }}

            IMPORTANT: Return ONLY the JSON object, no additional text."""

            # Get questions from LLM
            response = self._handle_api_call('questions', lambda client: 
                client.chat.completions.create(
                    model="llama-3.3-70b-specdec",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1024
                ).choices[0].message.content
            )
            
            return response

        except Exception as e:
            st.error(f"Error generating interview questions: {str(e)}")
            return None

    def improve_resume(self, areas=None, role=None):
        """Generate improvement suggestions using improvement-specific API key"""
        try:
            llm = self._get_llm('improvement')
            
            # Get weaknesses from analysis
            weaknesses = self.analysis_cache.get('weaknesses', []) if self.analysis_cache else []
            
            # Create prompt for improvements
            prompt = f"""Based on the following resume and analysis, provide specific improvement suggestions.

Original Resume:
{self.resume_text}

Role Requirements:
{self.role_requirements['requirements'] if self.role_requirements else 'General requirements'}

Previous Analysis:
{json.dumps(self.analysis_cache, indent=2)}

Identified Weaknesses:
{chr(10).join('- ' + w for w in weaknesses)}

Focus Areas: {areas if areas else 'All areas'}
Target Role: {role if role else self.role_requirements.get('role', 'Not specified')}

Generate improvement suggestions in this JSON format:
{{
    "content": {{
        "weaknesses_improvement": [
            {{
                "weakness": "specific weakness",
                "detailed_explanation": "explanation of why this is a weakness",
                "improvement_steps": [
                    "list of specific steps to improve this weakness"
                ]
            }}
        ],
        "skills": [
            "list of skill improvements"
        ],
        "experience": [
            "list of experience improvements"
        ],
        "education": [
            "list of education improvements"
        ],
        "format": [
            "list of formatting improvements"
        ]
    }}
}}

IMPORTANT: Return ONLY the JSON object, no additional text."""

            # Get improvements from LLM
            response = self._handle_api_call('improvement', lambda client: 
                client.chat.completions.create(
                    model="llama-3.3-70b-specdec",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1024
                ).choices[0].message.content
            )
            
            # Clean up the response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Try to parse as JSON
            try:
                improvements = json.loads(response)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract the JSON part
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        improvements = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        improvements = {
                            "content": {
                                "weaknesses_improvement": [
                                    {
                                        "weakness": "Error generating improvements",
                                        "detailed_explanation": "Please try again",
                                        "improvement_steps": []
                                    }
                                ],
                                "skills": ["Error generating improvements"],
                                "experience": [],
                                "education": [],
                                "format": []
                            }
                        }
                else:
                    raise ValueError("Could not parse improvements as JSON")
            
            return improvements

        except Exception as e:
            st.error(f"Error generating improvement suggestions: {str(e)}")
            return None

    def get_improved_resume(self, role=None, skills=None):
        """Generate an improved version of the resume using improved-resume-specific API key"""
        try:
            llm = self._get_llm('improved_resume')
            
            # Create prompt for improved resume
            prompt = f"""Based on the following resume and analysis, generate an improved version of the resume.

Original Resume:
{self.resume_text}

Role Requirements:
{self.role_requirements['requirements'] if self.role_requirements else 'General requirements'}

Target Role: {role if role else self.role_requirements.get('role', 'Not specified')}
Skills to Highlight: {', '.join(skills) if skills else 'All relevant skills'}

Previous Analysis:
{json.dumps(self.analysis_cache, indent=2)}

Generate an improved resume in this JSON format:
{{
    "sections": {{
        "professional_summary": "improved summary text",
        "skills": [
            "list of improved and reformatted skills"
        ],
        "experience": [
            {{
                "title": "job title",
                "company": "company name",
                "duration": "duration",
                "achievements": [
                    "list of improved bullet points"
                ]
            }}
        ],
        "education": [
            {{
                "degree": "degree name",
                "institution": "institution name",
                "duration": "duration",
                "details": [
                    "list of relevant details"
                ]
            }}
        ],
        "certifications": [
            {{
                "name": "certification name",
                "issuer": "issuer name",
                "date": "date"
            }}
        ]
    }}
}}

IMPORTANT: Return ONLY the JSON object, no additional text."""

            # Get improved resume from LLM
            response = self._handle_api_call('improved_resume', lambda client: 
                client.chat.completions.create(
                    model="llama-3.3-70b-specdec",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2048
                ).choices[0].message.content
            )
            
            # Clean up the response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Try to parse as JSON
            try:
                improved_resume = json.loads(response)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract the JSON part
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        improved_resume = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        improved_resume = {
                            "sections": {
                                "professional_summary": "Error generating improved resume",
                                "skills": ["Please try again"],
                                "experience": [],
                                "education": [],
                                "certifications": []
                            }
                        }
                else:
                    raise ValueError("Could not parse improved resume as JSON")
            
            return improved_resume

        except Exception as e:
            st.error(f"Error generating improved resume: {str(e)}")
            return None

    def cleanup(self):
        """Clean up temporary files"""
        try:
            if hasattr(self, 'improved_resume_path') and os.path.exists(self.improved_resume_path):
                os.unlink(self.improved_resume_path)
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")

# Example usage (e.g., in a Streamlit app or main script)
if __name__ == "__main__":
    # Test the agent
    agent = ResumeAnalysisAgent()
    result = agent.analyze_resume("path/to/resume.pdf")
    print(result)