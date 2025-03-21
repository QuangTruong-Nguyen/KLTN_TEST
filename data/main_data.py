from huggingFace_utils import *
from model_smry import *
from unstructured_utils import *
from vectorDB import *

# print("_______CHUNKING____________")
# chunks= chunk_data('D:/KLTN2/DM_C1.pdf')
# print("______Succcessec_chunking_____")


# print('________Saving Image_______')
# json_img=save_image(chunks)


# print("_________Text_______________")
# text_chunk=chunk_text(chunks)


# print("______Summary________________")
# text_summary=summarize_text(text_chunk)


embedddings=embedding()

# print("______Vector DB__________")

# setup_vector_store(text_summary, embedddings)

loaded_vector_store = Chroma(
    collection_name="example_collection2",
    embedding_function=embedddings,  # Phải sử dụng cùng embedding function
    persist_directory="./chroma_langchain_db"
)

test=loaded_vector_store.similarity_search(
    "Time-Series Data",
    k=2,
)

print(test)