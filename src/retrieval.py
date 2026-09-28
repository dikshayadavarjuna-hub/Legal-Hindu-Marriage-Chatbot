from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re


# 1. Read the extracted legal text
with open("data/hindu_marriage_act.txt", "r", encoding="utf-8") as file:
    text = file.read()


# 2. Divide the Act into sections
pattern = r"(?m)^\s*(\d{1,2})\.\s+[A-Z]"

matches = list(re.finditer(pattern, text))

chunks = []

for i, match in enumerate(matches):
    start = match.start()

    if i + 1 < len(matches):
        end = matches[i + 1].start()
    else:
        end = len(text)

    section = text[start:end].strip()

    if len(section) > 50:
        chunks.append(section)


print("Number of legal sections found:", len(chunks))


# 3. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 4. Convert legal sections into embeddings
embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")


# 5. Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


# 6. Search function
def search_legal_text(question, top_k=3):

    question_embedding = model.encode([question])

    question_embedding = np.array(question_embedding).astype("float32")

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results


# 7. Test question
question = "What are the conditions for a valid Hindu marriage?"

results = search_legal_text(question)


print("\nQuestion:", question)
print("\nMost relevant legal sections:\n")

for i, result in enumerate(results, start=1):
    print(f"--- Result {i} ---")
    print(result[:1500])
    print()