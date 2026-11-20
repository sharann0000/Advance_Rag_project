from src.config import retriever

question = "What is the company leave policy?"

documents = retriever.invoke(question)

print("Retrieved documents:", len(documents))

for i, doc in enumerate(documents, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)