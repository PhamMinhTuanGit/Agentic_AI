import os
import pdfplumber
import requests
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFEmbedder:
    def __init__(self, folder_path, model="nomic-embed-text", chunk_size=500, chunk_overlap=50):
        self.folder_path = folder_path
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.texts = []       # text chunks
        self.embeddings = []  # corresponding vectors

    def extract_text_from_pdf(self, file_path):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def get_embedding(self, text):
        response = requests.post(
            "http://ollama:11434/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_pdfs(self):
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(".pdf"):
                full_path = os.path.join(self.folder_path, filename)
                print(f"Processing: {filename}")
                raw_text = self.extract_text_from_pdf(full_path)
                if not raw_text.strip():
                    print(f"No text found in {filename}, skipping.")
                    continue
                chunks = self.text_splitter.split_text(raw_text)
                for chunk in chunks:
                    try:
                        embedding = self.get_embedding(chunk)
                        self.embeddings.append(embedding)
                        self.texts.append(chunk)
                    except Exception as e:
                        print(f"âŒ Error embedding chunk: {e}")
        return self.texts, self.embeddings

    def save_to_faiss(self, faiss_index_path="index.faiss", metadata_path="metadata.txt"):
        if not self.embeddings:
            print("âš ï¸ No embeddings to save.")
            return

        # Convert embeddings to NumPy array
        vectors = np.array(self.embeddings).astype("float32")

        # Create FAISS index
        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)

        # Save index to disk
        faiss.write_index(index, faiss_index_path)
        print(f"âœ… Saved FAISS index to {faiss_index_path}")

        # Save corresponding texts for reference
        with open(metadata_path, "w", encoding="utf-8") as f:
            for text in self.texts:
                f.write(text.strip().replace("\n", " ") + "\n")
        print(f"Saved metadata (text chunks) to {metadata_path}")

if __name__ == "__main__":
    embedder = PDFEmbedder(folder_path="./documents", model="nomic-embed-text")
    texts, vectors = embedder.embed_pdfs()
    embedder.save_to_faiss("docs_index.faiss", "docs_metadata.txt")

