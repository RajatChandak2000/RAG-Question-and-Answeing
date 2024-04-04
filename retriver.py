import argparse
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"
COSINE_SIMILARITY_THRESHOLD = 0.5  # Only consider documents with a cosine similarity above this value
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def compute_tfidf_cosine_similarity(query_text, documents):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents + [query_text])
    query_vector = tfidf_matrix[-1]
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix[:-1]).flatten()
    return cosine_similarities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print("Unable to find matching results.")
        return

    document_texts = [doc.page_content for doc, _score in results]
    tfidf_similarities = compute_tfidf_cosine_similarity(query_text, document_texts)

    # Filter documents based on TF-IDF cosine similarity threshold
    relevant_documents = [doc for doc, sim in zip(document_texts, tfidf_similarities) if sim > COSINE_SIMILARITY_THRESHOLD]

    # If no documents are sufficiently similar, fallback to the most relevant one
    if not relevant_documents:
        relevant_documents = [document_texts[0]]

    context_text_with_tfidf = "\n\n---\n\n".join(relevant_documents)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text_with_tfidf, question=query_text)


    context_text_chroma_search = "\n\n---\n\n".join(relevant_documents)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text_chroma_search, question=query_text)


    
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in zip(results, tfidf_similarities) if _score > COSINE_SIMILARITY_THRESHOLD]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
