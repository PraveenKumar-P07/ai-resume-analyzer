
import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

from utils.matcher import calculate_similarity
from utils.rag import index_resume, retrieve_resume_chunks


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

DEFAULT_MODEL = "openai/gpt-oss-20b"


# --------------------------------------------------
# 2. Groq helpers
# --------------------------------------------------

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    return Groq(api_key=api_key)


def get_groq_model():
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL)


def call_groq(messages, temperature=0.2):
    client = get_groq_client()

    response = client.chat.completions.create(
        model=get_groq_model(),
        messages=messages,
        temperature=temperature,
    )

    answer = response.choices[0].message.content

    if not answer:
        return "The AI did not return a response."

    return answer.strip()


# --------------------------------------------------
# 3. PDF resume extraction
# --------------------------------------------------

def extract_resume_text(uploaded_file):
    pdf_reader = PdfReader(BytesIO(uploaded_file.getvalue()))

    pages = []

    for page in pdf_reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)

    text = "\n".join(pages).strip()

    if not text:
        raise ValueError(
            "No readable text was found. "
            "If this is a scanned PDF, OCR may be required."
        )

    return text


# --------------------------------------------------
# 4. Skill-gap analysis
# --------------------------------------------------

def analyze_skill_gaps(resume_text, job_description):
    messages = [
        {
            "role": "system",
            "content": """
You are an expert resume analyst and career advisor.

Compare the supplied resume with the job description.

Return a structured report with these sections:
1. Matching Skills
2. Requirements Not Evidenced in the Resume
3. Relevant Strengths
4. Recommended Improvements
5. Suggested Learning and Portfolio Projects

Only use evidence from the supplied information.
Do not claim that a candidate has a skill or qualification
unless the resume supports it.

Clearly distinguish a genuinely missing skill from a skill
that is simply not mentioned in the resume.
"""
        },
        {
            "role": "user",
            "content": f"""
RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Prepare the skill-gap analysis.
"""
        },
    ]

    return call_groq(messages)


# --------------------------------------------------
# 5. Interview question generator
# --------------------------------------------------

def generate_interview_questions(
    resume_text,
    job_description,
    interview_type,
    question_count,
):
    messages = [
        {
            "role": "system",
            "content": """
You are an experienced technical interviewer and career coach.

Generate the exact requested number of interview questions.

For each question, include:
- The question
- What the interviewer evaluates
- A suggested sample answer
- A preparation tip

Use the resume and job description to personalize questions.

Never invent the candidate's projects, experience, or achievements.
For answers that require personal experience, use placeholders or
explain how the candidate should personalize the answer.
"""
        },
        {
            "role": "user",
            "content": f"""
Interview type: {interview_type}
Number of questions: {question_count}

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Generate the interview preparation report.
"""
        },
    ]

    return call_groq(messages, temperature=0.4)


# --------------------------------------------------
# 6. AI Resume Enhancer (NEW FEATURE)
# --------------------------------------------------

def enhance_resume(resume_text, job_description):
    messages = [
        {
            "role": "system",
            "content": """
You are an expert resume writer, ATS resume consultant,
and career coach.

Analyze the supplied resume against the supplied job description.

Create a practical resume improvement report with these sections:

1. PROFESSIONAL SUMMARY
Write an improved, concise summary tailored to the target role.
Use only qualifications and experience supported by the resume.

2. SKILLS TO HIGHLIGHT
Identify relevant skills already demonstrated or mentioned
in the resume. Explain why they matter for this role.

3. IMPROVED EXPERIENCE BULLET POINTS
Select existing resume bullet points and suggest clearer,
more impactful rewrites.

Preserve the original meaning. Do not invent employers,
responsibilities, tools, metrics, percentages, achievements,
certifications, or years of experience.

If a measurable result would strengthen a bullet point,
provide a clearly marked placeholder such as [actual result]
rather than making up a number.

If the resume contains no work experience, suggest improvements
to existing academic, internship, or personal project descriptions
where supported by the source.

4. ATS KEYWORD ANALYSIS
List relevant keywords and skills explicitly supported by the
resume, followed by relevant job-description keywords that are
not clearly evidenced in the resume.

Separate:
- Keywords already present
- Relevant keywords not found or not clearly evidenced
- Keywords that should only be added if genuinely applicable

Do not recommend keyword stuffing.

5. ACTION PLAN
Provide five prioritized and realistic improvements.

6. IMPORTANT NOTES
Identify any information the candidate should verify or fill in
before using the suggested content.

Be honest, specific, professional, and concise.
Do not guarantee an ATS score, interview, or job offer.
Do not treat an unmentioned skill as proof that the candidate
does not possess it.
"""
        },
        {
            "role": "user",
            "content": f"""
Please improve this resume for the target job.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Produce the complete resume improvement report.
"""
        },
    ]

    return call_groq(messages, temperature=0.2)


# --------------------------------------------------
# 7. Resume question answering using RAG
# --------------------------------------------------

def generate_resume_answer(question, passages):
    if not passages:
        return (
            "I could not retrieve relevant information from the "
            "indexed resume. Try rephrasing your question."
        )

    context = "\n\n".join(
        f"Resume passage {i + 1}:\n{item['text']}"
        for i, item in enumerate(passages)
    )

    messages = [
        {
            "role": "system",
            "content": """
You are a resume question-answering assistant.

Answer using only the supplied resume passages.
Do not invent facts or infer unsupported personal details.

If the passages do not contain enough information, say so.
Keep the answer clear and concise.
"""
        },
        {
            "role": "user",
            "content": f"""
RESUME PASSAGES:
{context}

QUESTION:
{question}

Answer using the resume passages only.
"""
        },
    ]

    return call_groq(messages)


# --------------------------------------------------
# 8. Session state
# --------------------------------------------------

defaults = {
    "resume_text": "",
    "resume_filename": "",
    "rag_ready": False,
    "chat_history": [],
    "skill_gap_report": "",
    "interview_report": "",
    "enhancement_report": "",
    "analyzed_job_description": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# 9. Application header
# --------------------------------------------------

st.title("📄 AI Resume Analyzer & Job Matcher")

st.write(
    "Analyze your resume, compare it with a job description, "
    "improve your resume, prepare for interviews, and ask "
    "questions about your resume."
)

st.caption(f"AI model: {get_groq_model()}")

st.divider()


# --------------------------------------------------
# 10. Upload resume
# --------------------------------------------------

st.header("1. Upload Your Resume")

uploaded_file = st.file_uploader(
    "Upload your resume in PDF format",
    type=["pdf"],
)

if uploaded_file is not None:
    if st.button("Extract Resume Text", type="primary"):
        try:
            with st.spinner("Extracting resume text..."):
                extracted_text = extract_resume_text(uploaded_file)

            st.session_state.resume_text = extracted_text
            st.session_state.resume_filename = uploaded_file.name

            # Reset features tied to the previous resume.
            st.session_state.rag_ready = False
            st.session_state.chat_history = []
            st.session_state.skill_gap_report = ""
            st.session_state.interview_report = ""
            st.session_state.enhancement_report = ""
            st.session_state.analyzed_job_description = ""

            st.success("Resume extracted successfully!")

        except Exception as error:
            st.error(f"Resume extraction failed: {error}")

if st.session_state.resume_text:
    st.success(
        f"Current resume: {st.session_state.resume_filename}"
    )

    with st.expander("View extracted resume text"):
        st.text_area(
            "Extracted text",
            value=st.session_state.resume_text,
            height=250,
            disabled=True,
        )

st.divider()


# --------------------------------------------------
# 11. Job description and similarity matching
# --------------------------------------------------

st.header("2. Resume and Job Matching")

job_description = st.text_area(
    "Paste the job description",
    height=220,
    placeholder=(
        "Paste the complete job description, including "
        "responsibilities, qualifications, and required skills."
    ),
    key="job_description_input",
)

if st.button("Analyze Resume and Job Match"):
    if not st.session_state.resume_text:
        st.warning("Please upload and extract a resume first.")

    elif not job_description.strip():
        st.warning("Please enter a job description.")

    else:
        try:
            with st.spinner("Calculating semantic similarity..."):
                similarity = calculate_similarity(
                    st.session_state.resume_text,
                    job_description,
                )

            st.session_state.analyzed_job_description = (
                job_description.strip()
            )

            st.session_state.skill_gap_report = ""
            st.session_state.interview_report = ""
            st.session_state.enhancement_report = ""

            st.subheader("Semantic Similarity")

            st.metric(
                "Resume and job description similarity",
                f"{similarity:.2f}%",
            )

            st.progress(
                min(max(similarity / 100, 0.0), 1.0)
            )

            st.info(
                "This percentage measures semantic similarity "
                "between the resume and job description. It is "
                "not a hiring probability, an ATS score, or a "
                "guarantee of job suitability."
            )

        except Exception as error:
            st.error(f"Matching failed: {error}")


st.divider()


# --------------------------------------------------
# 12. AI skill-gap analysis
# --------------------------------------------------

st.header("3. AI Job Skill-Gap Analysis")

if st.button("Generate Skill-Gap Report"):
    if not st.session_state.resume_text:
        st.warning("Please upload and extract a resume first.")

    elif not job_description.strip():
        st.warning("Please enter a job description first.")

    else:
        try:
            with st.spinner("Analyzing skills and requirements..."):
                report = analyze_skill_gaps(
                    st.session_state.resume_text,
                    job_description,
                )

            st.session_state.skill_gap_report = report
            st.session_state.analyzed_job_description = (
                job_description.strip()
            )

        except Exception as error:
            st.error(f"Skill-gap analysis failed: {error}")

if st.session_state.skill_gap_report:
    st.markdown(st.session_state.skill_gap_report)

    st.download_button(
        "Download Skill-Gap Report",
        data=st.session_state.skill_gap_report,
        file_name="skill_gap_report.md",
        mime="text/markdown",
    )


st.divider()


# --------------------------------------------------
# 13. AI Resume Enhancer (NEW)
# --------------------------------------------------

st.header("4. AI Resume Enhancer")

st.write(
    "Get targeted suggestions for your professional summary, "
    "experience bullet points, relevant skills, and ATS keywords."
)

st.warning(
    "Review every suggestion before using it. The AI must not "
    "add skills, experience, or achievements that you cannot "
    "honestly support."
)

if st.button(
    "✨ Generate Resume Improvement Report",
    type="primary",
):
    if not st.session_state.resume_text:
        st.warning("Please upload and extract a resume first.")

    elif not job_description.strip():
        st.warning(
            "Please paste the job description before enhancing "
            "your resume."
        )

    else:
        try:
            with st.spinner(
                "Generating personalized resume improvements..."
            ):
                report = enhance_resume(
                    st.session_state.resume_text,
                    job_description,
                )

            st.session_state.enhancement_report = report
            st.session_state.analyzed_job_description = (
                job_description.strip()
            )

        except Exception as error:
            st.error(f"Resume enhancement failed: {error}")

if st.session_state.enhancement_report:
    st.subheader("Your Resume Improvement Report")

    st.markdown(st.session_state.enhancement_report)

    st.download_button(
        "⬇️ Download Resume Improvement Report",
        data=st.session_state.enhancement_report,
        file_name="resume_improvement_report.md",
        mime="text/markdown",
    )


st.divider()


# --------------------------------------------------
# 14. Interview question generator
# --------------------------------------------------

st.header("5. AI Interview Question Generator")

col1, col2 = st.columns(2)

with col1:
    interview_type = st.selectbox(
        "Interview type",
        ["Technical", "HR", "Both"],
    )

with col2:
    question_count = st.selectbox(
        "Number of questions",
        [5, 10, 15],
    )

if st.button("Generate Interview Questions"):
    if not st.session_state.resume_text:
        st.warning("Please upload and extract a resume first.")

    elif not job_description.strip():
        st.warning("Please enter a job description first.")

    else:
        try:
            with st.spinner("Preparing interview questions..."):
                report = generate_interview_questions(
                    st.session_state.resume_text,
                    job_description,
                    interview_type,
                    question_count,
                )

            st.session_state.interview_report = report
            st.session_state.analyzed_job_description = (
                job_description.strip()
            )

        except Exception as error:
            st.error(f"Interview generation failed: {error}")

if st.session_state.interview_report:
    st.markdown(st.session_state.interview_report)

    st.download_button(
        "Download Interview Preparation Report",
        data=st.session_state.interview_report,
        file_name="interview_preparation_report.md",
        mime="text/markdown",
    )


st.divider()


# --------------------------------------------------
# 15. Index resume for RAG
# --------------------------------------------------

st.header("6. Prepare Resume for AI Questions")

st.write(
    "Index the resume so the application can retrieve relevant "
    "passages when you ask questions about its contents."
)

if st.button("Index Resume for RAG"):
    if not st.session_state.resume_text:
        st.warning("Please upload and extract a resume first.")

    else:
        try:
            with st.spinner("Indexing resume passages..."):
                chunk_count = index_resume(
                    st.session_state.resume_text
                )

            st.session_state.rag_ready = True
            st.session_state.chat_history = []

            st.success(
                f"Resume indexed successfully. "
                f"Created {chunk_count} text chunks."
            )

        except Exception as error:
            st.session_state.rag_ready = False
            st.error(f"Resume indexing failed: {error}")


st.divider()


# --------------------------------------------------
# 16. Chat with your resume
# --------------------------------------------------

st.header("7. Chat with Your Resume")

st.write(
    "Ask questions about skills, education, projects, "
    "certifications, or experience mentioned in your resume."
)

if not st.session_state.rag_ready:
    st.info(
        "Please click 'Index Resume for RAG' before asking "
        "questions."
    )

else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about your resume..."
    )

    if question:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "Retrieving relevant resume passages..."
                ):
                    passages = retrieve_resume_chunks(
                        question,
                        top_k=3,
                    )

                if not passages:
                    answer = (
                        "I could not retrieve relevant information "
                        "from the indexed resume. Please try "
                        "rephrasing your question."
                    )
                else:
                    answer = generate_resume_answer(
                        question,
                        passages,
                    )

                st.markdown(answer)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as error:
                st.error(f"Resume Q&A failed: {error}")


# --------------------------------------------------
# 17. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI Resume Analyzer & Job Matcher | "
    "Python · Streamlit · Sentence Transformers · "
    "ChromaDB · Groq"
)
