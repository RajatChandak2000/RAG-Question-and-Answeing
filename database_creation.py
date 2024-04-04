import os
import shutil
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

# Define constants for paths and settings
CHROMA_PATH = "chroma"
DATA_PATH = "dataset/Alice in wonderland.txt"
GLOB_PATTERN = "*.md"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100


def initialize_chroma_db(chroma_path: str):
    """Remove existing Chroma database directory if it exists and create a new one."""
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
    os.makedirs(chroma_path)


def load_markdown_documents(data_path: str, glob_pattern: str):
    """Load markdown documents from a specified directory using a glob pattern."""
    loader = DirectoryLoader(data_path, glob=glob_pattern)
    return loader.load()


def split_documents_into_chunks(documents, chunk_size: int, chunk_overlap: int):
    """Split the loaded documents into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


def display_document_sample(chunks, sample_index: int):
    """Display a sample chunk to check the content and metadata."""
    document = chunks[sample_index]
    print(f"Sample document content: {document.page_content}")
    print(f"Sample document metadata: {document.metadata}")


def persist_chunks_to_chroma(chunks, embeddings_model, chroma_path: str):
    """Save the document chunks to a Chroma vector store."""
    db = Chroma.from_documents(chunks, embeddings_model, persist_directory=chroma_path)
    db.persist()


def main():
    initialize_chroma_db(CHROMA_PATH)
    documents = load_markdown_documents(DATA_PATH, GLOB_PATTERN)
    chunks = split_documents_into_chunks(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    display_document_sample(chunks, 10)
    persist_chunks_to_chroma(chunks, OpenAIEmbeddings(), CHROMA_PATH)


if __name__ == "__main__":
    main()
