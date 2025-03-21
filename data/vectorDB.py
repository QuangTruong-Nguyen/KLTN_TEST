from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4
from typing import List


def setup_vector_store(text_summary: List[str], embeddings) -> None:
    vector_store = Chroma(
        collection_name="example_collection2",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )

    document_list = [Document(page_content=text, id=i) for i, text in enumerate(text_summary)]
    uuids = [str(uuid4()) for _ in range(len(document_list))]

    vector_store.add_documents(documents=document_list, ids=uuids)