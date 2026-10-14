
from utils.rag import index_resume, retrieve_resume_chunks

sample_resume = """
Alex Morgan is a junior Python developer.

Technical skills include Python, SQL, Streamlit,
Generative AI fundamentals, embeddings, and cosine similarity.

Alex built an AI Resume Analyzer prototype using Streamlit
and PyPDF to extract text from resume documents.

Alex also practiced semantic text similarity using
Sentence Transformers and cosine similarity.

Alex has basic knowledge of REST APIs, Git, and GitHub.
"""

# Step 1: Store resume chunks in ChromaDB
chunk_count = index_resume(sample_resume)

print(f"Resume indexed successfully: {chunk_count} chunks")

# Step 2: Search the resume
question = "Which AI and Python projects has Alex built?"

results = retrieve_resume_chunks(question)

print("\nRelevant resume passages:\n")

for item in results:
    print("-", item["text"])
    print()
