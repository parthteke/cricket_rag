import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Load cricket facts
df = pd.read_csv("cricket_rag.csv")
cricket_facts = df['fact'].tolist()

# Load model and create normalized embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(cricket_facts, convert_to_numpy=True)

# Normalize embeddings to unit vectors
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Build FAISS index using Inner Product for cosine similarity
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Answer function using cosine similarity
def answer_query(query, similarity_threshold=0.45):
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    similarities, indices = index.search(query_embedding, k=1)
    similarity = float(similarities[0][0])

    print(f"Cosine Similarity: {similarity:.4f}")

    if similarity >= similarity_threshold:
        return cricket_facts[indices[0][0]]
    else:
        return "I'm not able to answer that question."

# Main loop
print("\n🏏 Cricket Q&A Bot (Cosine-based) — type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.strip().lower() == "exit":
        break
    response = answer_query(query)
    print("Bot:", response)


