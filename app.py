from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load document
with open("data/policy.txt", "r", encoding="utf-8") as file:
    document_text = file.read()

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

# Split into chunks
chunks = text_splitter.split_text(document_text)

print(f"Total chunks created: {len(chunks)}")
print("Chunks:")
for i, chunk in enumerate(chunks, start=1):
    print(f"{i}. {chunk}")
