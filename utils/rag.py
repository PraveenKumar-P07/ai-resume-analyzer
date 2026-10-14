
import chromadb
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a persistent ChromaDB database
client = chromadb.PersistentClient(path="./chroma_db")


def split_into_chunks(text, chunk_size=500, overlap=100):
    """Split resume text into overlapping chunks."""

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


def index_resume(resume_text):
    """Create embeddings and store resume chunks in ChromaDB."""

    chunks = split_into_chunks(resume_text)

    if not chunks:
        raise ValueError("No readable resume text to index.")

    # Replace the previous indexed resume
    try:
        client.delete_collection("resume_chunks")
    except Exception:
        pass

    collection = client.create_collection(
        name="resume_chunks",
        metadata={"hnsw:space": "cosine"},
    )

    embeddings = model.encode(chunks).tolist()

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"chunk_index": i}
            for i in range(len(chunks))
        ],
    )

    return len(chunks)


def retrieve_resume_chunks(question, top_k=3):
    """Retrieve resume passages most relevant to a question."""

    question = question.strip()

    if not question:
        return []

    collection = client.get_collection("resume_chunks")

    count = collection.count()

    if count == 0:
        return []

    question_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, count),
        include=["documents", "distances"],
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    return [
        {"text": document, "distance": distance}
        for document, distance in zip(documents, distances)
    ]
