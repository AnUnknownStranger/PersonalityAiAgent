from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag.config import VECTOR_DB_PATH, EMBEDDING_MODEL, TOP_K

_db = None


def _load_db():
    global _db

    if _db is None:
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        _db = FAISS.load_local(
            VECTOR_DB_PATH,
            embedding,
            allow_dangerous_deserialization=True
        )

    return _db


def retrieve_context(query: str):
    """

    Inputs:
        query (str): user query

    Outputs:
        List[str]: related texts (Top-K)
    """
    db = _load_db()

    docs = db.similarity_search(query, k=TOP_K)

    results = [d.page_content for d in docs]

    return results

''' Test code '''
if __name__ == "__main__":
    while True:
        q = input("\nQuery: ")
        results = retrieve_context(q)

        print("\nTop Results:")
        for i, r in enumerate(results):
            print(f"{i+1}. {r}")
