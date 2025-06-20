# main.py

import fitz  # PyMuPDF
import gradio as gr
from resume_chain import load_llm, get_resume_chain

# Read text from uploaded PDF
def read_pdf(file):
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        return f"❌ Error reading PDF: {str(e)}"

# Gradio interface function
def review_resume(pdf_file):
    resume_text = read_pdf(pdf_file)
    if resume_text.startswith("❌"):
        return resume_text

    llm = load_llm()
    chain = get_resume_chain(llm)
    output = chain.run(resume_text)
    return output

# UI
gr.Interface(
    fn=review_resume,
    inputs=gr.File(label="📄 Upload your Resume (PDF)"),
    outputs=gr.Textbox(label="📝 AI Review Output", lines=20),
    title="AI Resume Reviewer (Offline LLaMA)",
    description="Get strengths, weaknesses, suggestions, and ATS keywords from your resume using an offline LLM.",
).launch()
