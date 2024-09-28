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
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

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

            Ensure that the email blends the candidate's skills and experience with the job's requirements effectively, showing why they are a strong fit.

            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "resume_text": resume_text})
        return res.content
