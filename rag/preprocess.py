import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.config import DATA_PATH


def load_data():
    texts = []

    for file in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, file)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append({
                        "text": line,
                        "movie": file
                    })

    return texts


def chunk_data(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    docs = []

    for item in data:
        chunks = splitter.split_text(item["text"])

        for chunk in chunks:
            docs.append({
                "page_content": chunk,
                "metadata": {
                    "movie": item["movie"],
                    "character": "Harry Potter"
                }
            })

    return docs


def run_preprocess():
    data = load_data()
    docs = chunk_data(data)
    return docs