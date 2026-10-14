# AI Resume Analyzer & Job Matcher

An AI-powered application that analyzes resumes, compares them with job descriptions, identifies skill gaps, improves resume content, generates interview questions, and answers questions about the uploaded resume.

## Features

- **Resume Upload:** Upload a resume in PDF format.
- **Resume and Job Matching:** Compare a resume with a job description using semantic similarity.
- **AI Skill Gap Analysis:** Identify missing or relevant skills for a target role.
- **AI Resume Enhancer:** Generate suggestions to improve resume content.
- **Interview Question Generator:** Generate interview questions based on the resume and job description.
- **Resume Q&A:** Ask questions about the uploaded resume using Retrieval-Augmented Generation (RAG).
- **Downloadable Reports:** Download generated resume enhancement results where supported by the application.

## Technologies Used

- Python
- Streamlit
- Groq API
- Sentence Transformers
- Scikit-learn
- ChromaDB
- PyPDF
- LangChain Text Splitters
- python-dotenv

## Project Structure

```text
resume-ai-analyzer/
├── app.py
├── app_backup.py
├── requirements.txt
├── .gitignore
├── README.md
├── test_env.py
├── test_matcher.py
├── test_rag.py
└── utils/
    ├── matcher.py
    └── rag.py
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/PraveenKumar-P07/ai-resume-analyzer.git
cd ai-resume-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a file named `.env` in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Replace `your_groq_api_key` with your own Groq API key. Never upload your real API key to GitHub.

### 6. Run the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL that you can open in your browser.

## How It Works

1. Upload a resume in PDF format.
2. Enter or provide a job description.
3. Generate a semantic similarity score to compare the resume with the job description.
4. Review AI-generated skill gap analysis and resume improvement suggestions.
5. Generate interview questions.
6. Ask questions about the uploaded resume through the RAG-based chat feature.

## Important Notes

- The similarity score represents semantic similarity between the resume and job description. It is not a hiring probability or a guaranteed ATS score.
- AI-generated suggestions may contain errors. Review them before using them in an actual job application.
- Results depend on the quality of the resume, job description, extracted text, and AI-generated responses.
- Keep API keys and other secrets in your local `.env` file. Do not publish them.
- The RAG functionality is designed around the resume currently indexed by the application.

## Future Improvements

- Support additional document formats.
- Add more detailed skill extraction and matching.
- Improve evaluation and testing.
- Add support for multiple resumes and job descriptions.
- Improve the user interface and reporting features.

## Author

**Praveen Kumar**

GitHub: https://github.com/PraveenKumar-P07

## License

No license has been specified yet.