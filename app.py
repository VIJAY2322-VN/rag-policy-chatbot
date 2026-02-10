from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Load document
with open("data/policy.txt", "r", encoding="utf-8") as file:
    document_text = file.read()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
chunks = text_splitter.split_text(document_text)

print(f"Total chunks: {len(chunks)}")

# 3. Create embeddings (FREE, local)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store in FAISS vector database
vector_db = FAISS.from_texts(chunks, embedding_model)

print("FAISS vector store created successfully")

# 5. Similarity search
query = "What is the work from home policy?"
results = vector_db.similarity_search(query, k=2)

print("\nQuery:", query)
print("Relevant chunks:")
for i, doc in enumerate(results, start=1):
    print(f"{i}. {doc.page_content}")
