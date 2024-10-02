import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills`, `description`, and `job_id` (if available).
            Only return one valid JSON job object, selecting the best match.
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, dict) else res[0]  # Return only one job

    def write_mail(self, job, resume_text):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### CANDIDATE'S RESUME:
            {resume_text}

            ### INSTRUCTION:
            You are the candidate applying for the above job. Write a professional cold email to the hiring manager, convincing them to offer you an interview opportunity. 
            The email should:
            - Be concise, professional, and to the point (3-4 paragraphs max).
            - Focus on the skills required for the job and emphasize the candidate's matching skills from their resume.
            - Highlight relevant projects, experience, and achievements from the resume that align with the job description.
            - Reference specific requirements from the job description, such as experience level or key skills, and show how the candidate fulfills them.
            - Avoid unnecessary preambles and get straight to the point.

            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "resume_text": resume_text})
        return res.content

    def write_referral_message(self, job, resume_text, name=""):
        prompt_referral = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### CANDIDATE'S RESUME:
            {resume_text}

            ### INSTRUCTION:
            You are a candidate seeking a referral from an existing employee for the job mentioned above. Write a professional and concise message asking for a referral for the specific job role, incorporating the following:
            - Start by clearly stating that you are interested in the job role and mention the job ID (if available).
            - Briefly introduce your key skills that match the job requirements (1-2 sentences).
            - Provide a short overview of your experience (1-2 sentences).
            - Optionally, if applicable, mention any problem-solving experience on platforms like LeetCode (if you have solved a significant number of problems).
            - Close with a polite request for a referral.

            The email should be short, professional, and respectful of the employee's time, while showcasing your qualifications effectively.

            ### REFERRAL MESSAGE (NO PREAMBLE):
            """
        )
        job_description = f"Job Title: {job['role']}, Job ID: {job.get('job_id', 'N/A')}"
        chain_referral = prompt_referral | self.llm
        # Ensure resume_text is passed in the input dictionary
        res = chain_referral.invoke({"job_description": job_description, "resume_text": resume_text, "name": name})
        return res.content
