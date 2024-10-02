import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from utils import clean_text, parse_pdf_resume

def create_streamlit_app(llm, clean_text):
    st.title("📧 Cold Mail & Referral Generator")

    # Job URL input
    url_input = st.text_input("Enter a job URL:", value="")

    # Resume PDF upload input
    resume_file = st.file_uploader("Upload your resume (PDF only)", type="pdf")

    # Select between email to hiring manager or referral message
    email_type = st.selectbox("Choose the type of message:", ["Email to Hiring Manager", "Referral Request to Employee"])

    if email_type == "Referral Request to Employee":
        employee_name = st.text_input("Employee Name (optional):")

    submit_button = st.button("Submit")

    if submit_button:
        try:
            if url_input and resume_file is not None:
                resume_text = parse_pdf_resume(resume_file)
                st.text("Resume uploaded successfully!")
                st.text("Please wait...")

                # Load and process job posting data from the provided URL
                loader = WebBaseLoader([url_input])
                job_data = clean_text(loader.load().pop().page_content)

                # Extract job details using LLM
                job = llm.extract_jobs(job_data)

                if email_type == "Email to Hiring Manager":
                    # Generate email to the hiring manager
                    email = llm.write_mail(job, resume_text)
                    st.code(email, language='markdown')
                else:
                    # Generate referral message to the employee
                    referral_message = llm.write_referral_message(job, resume_text, name=employee_name)
                    st.code(referral_message, language='markdown')

            else:
                if not url_input:
                    st.error("Please enter the job URL.")
                if resume_file is None:
                    st.error("Please upload your resume.")
                
        except Exception as e:
            st.error(f"An Error Occurred: {e}")

if __name__ == "__main__":
    chain = Chain()
    st.set_page_config(layout="wide", page_title="Cold Email & Referral Generator", page_icon="📧")
    create_streamlit_app(chain, clean_text)
