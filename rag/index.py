from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag.preprocess import run_preprocess
from rag.config import VECTOR_DB_PATH, EMBEDDING_MODEL
import os


def build_index():
    docs = run_preprocess()

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"local_files_only": True}
    )

    texts = [d["page_content"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    db = FAISS.from_texts(texts, embedding, metadatas=metadatas)

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    db.save_local(VECTOR_DB_PATH)

    print("Vector DB built.")


if __name__ == "__main__":
    build_index()
