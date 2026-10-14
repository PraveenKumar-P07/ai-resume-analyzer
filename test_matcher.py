
from utils.matcher import calculate_similarity

resume = """
Python developer with SQL knowledge.
Built a Streamlit application and experimented
with embeddings and semantic similarity.
"""

job_description = """
Looking for a Python developer with SQL skills,
AI application development experience,
and knowledge of semantic search.
"""

score = calculate_similarity(resume, job_description)

print(f"Resume Similarity Score: {score}%")
