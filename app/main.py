import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from utils import clean_text, parse_pdf_resume

def create_streamlit_app(llm, clean_text):
    st.title("📧 Cold Mail Generator")

    # Job URL input (remove the hardcoded value)
    url_input = st.text_input("Enter a job URL:", value="")

    # Resume PDF upload input
    resume_file = st.file_uploader("Upload your resume (PDF only)", type="pdf")

    submit_button = st.button("Submit")

    if submit_button:
        try:
            # Check if both URL and resume are provided
            if url_input and resume_file is not None:
                # Parse the resume PDF
                resume_text = parse_pdf_resume(resume_file)
                st.text("Resume uploaded successfully!")
                st.text("Please wait")

                # Load and process the job posting data from the provided URL
                loader = WebBaseLoader([url_input])
                job_data = clean_text(loader.load().pop().page_content)

                # Extract job details using LLM
                jobs = llm.extract_jobs(job_data)

                for job in jobs:
                    # Generate email using the job and the resume text
                    email = llm.write_mail(job, resume_text)
                    st.code(email, language='markdown')

            else:
                # Check which part is missing and show an error
                if not url_input:
                    st.error("Please enter the job URL.")
                if resume_file is None:
                    st.error("Please upload your resume.")

        except Exception as e:
            st.error(f"An Error Occurred: {e}")

if __name__ == "__main__":
    chain = Chain()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, clean_text)
