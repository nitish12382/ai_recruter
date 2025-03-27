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
    def __init__(self, api_key):
        """Initialize the ResumeAnalysisAgent with the provided API key"""
        self.api_key = api_key
        self.client = Groq(api_key=api_key)
        self.vector_store = None
        self.resume_text = None
        self.role_requirements = None
        self.llm = GroqLLM(client=self.client)
        self.cutoff_score = 75
        self.resume_weaknesses = []
        self.resume_strengths = []
        self.improvement_suggestions = {}
        self.jd_text = None
        self.extracted_skills = None

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

            response = self.llm(prompt)
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

            response = self.llm(prompt)
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
            llm=self.llm,
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

    def analyze_resume(self, resume_file, custom_jd=None):
        """Analyze a resume and return the results"""
        try:
            # Extract text from PDF
            self.resume_text = self.extract_text_from_pdf(resume_file)
            
            # Create vector store for RAG
            self.vector_store = self.create_rag_vector_store(self.resume_text)
            
            # Get role requirements
            if custom_jd:
                if isinstance(custom_jd, dict):
                    self.role_requirements = custom_jd
                else:
                    # If custom_jd is a string (file path), extract text from it
                    self.role_requirements = {
                        "role": "Custom Role",
                        "requirements": self.extract_text_from_file(custom_jd)
                    }
            else:
                self.role_requirements = {
                    "role": "General Role",
                    "requirements": "General requirements for resume analysis"
                }
            
            # Analyze resume
            analysis_prompt = f"""Analyze this resume for the role of {self.role_requirements['role']}:

Resume:
{self.resume_text}

Role Requirements:
{self.role_requirements['requirements']}

Provide a detailed analysis in the following JSON format:
{{
    "score": <number between 0 and 100>,
    "strengths": [<list of strengths>],
    "weaknesses": [<list of weaknesses>],
    "improvement_suggestions": {{
        "technical": [<list of suggestions>],
        "experience": [<list of suggestions>],
        "education": [<list of suggestions>]
    }},
    "ats_keywords": [<list of keywords>]
}}

IMPORTANT: Respond ONLY with the JSON object, no additional text or explanations."""

            # Get analysis from LLM
            response = self.llm.invoke(analysis_prompt)
            
            # Filter out the think section if present
            if "<think>" in response:
                response = response.split("</think>")[-1].strip()
            
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
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract the JSON part
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        analysis_result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        # If still failing, create a default result
                        analysis_result = {
                            "score": 0,
                            "strengths": ["Unable to analyze strengths"],
                            "weaknesses": ["Unable to analyze weaknesses"],
                            "improvement_suggestions": {
                                "technical": ["Unable to generate technical suggestions"],
                                "experience": ["Unable to generate experience suggestions"],
                                "education": ["Unable to generate education suggestions"]
                            },
                            "ats_keywords": ["Unable to extract keywords"]
                        }
                else:
                    raise ValueError("Could not parse analysis result as JSON")
            
            # Store results
            self.resume_weaknesses = analysis_result.get('weaknesses', [])
            self.resume_strengths = analysis_result.get('strengths', [])
            self.improvement_suggestions = analysis_result.get('improvement_suggestions', {})
            
            return analysis_result
            
        except Exception as e:
            raise Exception(f"Error analyzing resume: {str(e)}")

    def ask_question(self, question):
        """Ask a question about the resume"""
        if not self.vector_store or not self.resume_text:
            return "Please analyze a resume first."
            
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        
        response = qa_chain.run(question)
        return response

    def generate_interview_questions(self, question_types, difficulty, num_questions):
        """Generate interview questions based on the resume"""
        if not self.resume_text:
            return []
            
        try:
            # Get the analysis result if not already available
            if not hasattr(self, 'analysis_result'):
                self.analysis_result = self.analyze_resume(self.resume_file_path)
            
            # Get skills from the analysis result
            skills = self.analysis_result.get('ats_keywords', [])
            strengths = self.analysis_result.get('strengths', [])
            weaknesses = self.analysis_result.get('weaknesses', [])
            
            context = f"""
            Resume Content:
            {self.resume_text[:2000]} ...
            Skills to focus on: {', '.join(skills)}
            Strengths: {', '.join(strengths)}
            Areas for improvement: {', '.join(weaknesses)}
            """
            
            prompt = f"""
            Generate {num_questions} personalized {difficulty.lower()} level interview questions for this candidate
            based on their resume and skills. Include only the following question types: {', '.join(question_types)}.

            For each question:
            1. Clearly label the question type
            2. Make the question specific to their background and skills
            3. For coding questions, include a clear problem statement

            {context}
            Format the response as a list of tuples with the question type and the question itself.
            Each tuple should be in the format: ("Question Type", "Full Question Text")
            """
            
            response = self.llm(prompt)
            questions_text = response
            
            questions = []
            pattern = r'[("]([^"]+)[",)\s]+[(",\s]+([^"]+)[")\s]+'
            matches = re.findall(pattern, questions_text, re.DOTALL)
            
            for match in matches:
                if len(match) >= 2:
                    question_type = match[0].strip()
                    question = match[1].strip()
                    
                    for requested_type in question_types:
                        if requested_type.lower() in question_type.lower():
                            questions.append((requested_type, question))
                            break
            
            if not questions:
                lines = questions_text.split('\n')
                current_type = None
                current_question = ""
                
                for line in lines:
                    line = line.strip()
                    if any(t.lower() in line.lower() for t in question_types) and not current_question:
                        current_type = next((t for t in question_types if t.lower() in line.lower()), None)
                    if ":" in line:
                        current_question = line.split(":", 1)[1].strip()
                    elif current_type and line:
                        current_question += " " + line
                    elif current_type and current_question:
                        questions.append((current_type, current_question))
                        current_type = None
                        current_question = ""
            
            questions = questions[:num_questions]
            return questions
            
        except Exception as e:
            print(f"Error generating interview questions: {e}")
            return []

    def improve_resume(self, improvement_areas, target_role=""):
        """Generate suggestions to improve the resume"""
        if not self.resume_text:
            return {}
            
        try:
            improvements = {}
            
            # Get the analysis result if not already available
            if not hasattr(self, 'analysis_result'):
                self.analysis_result = self.analyze_resume(self.resume_file_path)
            
            # Generate improvements for each area
            for area in improvement_areas:
                if area == "Skills Highlighting":
                    improvements[area] = {
                        "description": "Your resume needs to better highlight key skills that are important for the role.",
                        "specific": []
                    }
                    
                    # Add specific suggestions based on weaknesses
                    for weakness in self.resume_weaknesses:
                        if isinstance(weakness, dict):
                            skill_name = weakness.get("skill", "")
                            if "suggestions" in weakness and weakness["suggestions"]:
                                for suggestion in weakness["suggestions"]:
                                    improvements[area]["specific"].append(f"{skill_name}: {suggestion}")
                        else:
                            improvements[area]["specific"].append(str(weakness))
                
                elif area == "Achievement Quantification":
                    improvements[area] = {
                        "description": "Your resume should include more quantifiable achievements and metrics.",
                        "specific": [
                            "Add specific numbers and percentages to your achievements",
                            "Include metrics like revenue growth, cost savings, or performance improvements",
                            "Quantify the impact of your work on business outcomes"
                        ]
                    }
                
                elif area == "Experience Relevance":
                    improvements[area] = {
                        "description": "Your experience section could be more relevant to the target role.",
                        "specific": [
                            "Focus on experiences that align with the role requirements",
                            "Highlight relevant projects and their outcomes",
                            "Emphasize transferable skills and achievements"
                        ]
                    }
                
                elif area == "Education Focus":
                    improvements[area] = {
                        "description": "Your education section could be more focused on relevant qualifications.",
                        "specific": [
                            "Highlight relevant coursework and academic achievements",
                            "Include relevant certifications and training",
                            "Emphasize research or projects related to the role"
                        ]
                    }
                
                elif area == "Professional Summary":
                    improvements[area] = {
                        "description": "Your professional summary could be more impactful.",
                        "specific": [
                            "Start with a strong opening statement",
                            "Highlight key achievements and skills",
                            "Align with the target role's requirements"
                        ]
                    }
            
            return improvements
            
        except Exception as e:
            print(f"Error generating resume improvements: {e}")
            # Return default improvements for each area
            return {
                area: {
                    "description": f"Improvements needed in {area}",
                    "specific": ["Review and enhance this section"]
                } for area in improvement_areas
            }
        
    def get_improved_resume(self, target_role="", highlight_skills=""):
        """Get an improved version of the resume"""
        try:
            if not self.resume_text:
                return ""
                
            # Get the analysis result if not already available
            if not hasattr(self, 'analysis_result'):
                self.analysis_result = self.analyze_resume(self.resume_file_path)
                
            weaknesses_text = ""
            for weakness in self.resume_weaknesses:
                if isinstance(weakness, dict):
                    skill_name = weakness.get("skill", "")
                    if "suggestions" in weakness and weakness["suggestions"]:
                        for suggestion in weakness["suggestions"]:
                            weaknesses_text += f"For {skill_name}: {suggestion}\n"
                else:
                    weaknesses_text += f"{weakness}\n"
                    
            context = f"""
            Resume Content:
            {self.resume_text[:2000]} ...
            Strengths: {', '.join(self.resume_strengths)}
            Areas for improvement: {', '.join(self.analysis_result.get('weaknesses', []))}
            Weaknesses: {weaknesses_text}
            """
            
            prompt = f"""
            Generate an improved version of the resume optimized for the role of {target_role}. 
            Highlight the following skills: {highlight_skills}. 
            Include specific suggestions for improvement and formatting. 
            Format the resume in a modern, clean style with clear section headings.
            {context}
            """
            
            response = self.llm(prompt)
            
            # Create a temporary file for the improved resume
            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as tmp:
                tmp.write(response)
                self.improved_resume_path = tmp.name
                
            return self.improved_resume_path
            
        except Exception as e:
            print(f"Error generating improved resume: {e}")
            return ""

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
    agent = ResumeAnalysisAgent("your-api-key-here")
    result = agent.analyze_resume("path/to/resume.pdf")
    print(result)