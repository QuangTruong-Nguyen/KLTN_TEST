import warnings
warnings.filterwarnings('ignore')
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
# from unstructured.staging.base import dict_to_elements, elements_to_json
from IPython.display import JSON
import os
import json
from unstructured.partition.api import partition_via_api
import base64
from IPython.display import Image, display
from typing import Any,List, Dict


# secret_value_0 = 
# secret_value_1 = 
# s = UnstructuredClient(
#     api_key_auth=secret_value_0,
#     server_url=secret_value_1,
# )


def save_image(chunks: List[Any]) -> List[Dict[str, Any]]:
    img_des = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for j in range(len(chunk_els)):
                if "Image" in str(type(chunk_els[j])):
                    # Ensure there is a next element to avoid index errors
                    if j + 1 < len(chunk_els):
                        img_des.append({
                            "img": chunk_els[j].metadata.image_base64,
                            "text": chunk_els[j + 1].to_dict()['text']
                        })
    return img_des



def get_images_base64(chunks: List[Any]) -> List[str]:
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
                    
    return images_b64

def chunk_data(path_file: str) -> Any:

    chunks = partition_via_api(
        filename=path_file,
        infer_table_structure=True,            
        strategy="hi_res",                     

        extract_image_block_types=["Image"],   
        extract_image_block_to_payload=True,   
        chunking_strategy="by_title",          
        max_characters=10000,                  
        combine_text_under_n_chars=2000,       
        new_after_n_chars=6000,

        # api_url=,
        # api_key=,
    )
    return chunks


def chunk_text(chunks: List[Any]) -> List[str]:
    
    """
    Trích xuất và định dạng văn bản từ các đoạn đã chia nhỏ.

    Args:
        chunks (List[Any]): Danh sách các đoạn văn bản đã được chia nhỏ.

    Returns:
        List[str]: Danh sách các đoạn văn bản đã được định dạng.
    """
    text_chunks = []

    for chunk in chunks:
        chunk_texts = []
        chunk_elements = getattr(chunk.metadata, "orig_elements", [])

        for el in chunk_elements:
            el_data = el.to_dict()
            el_type = el_data.get("type", "")
            el_text = el_data.get("text", "")

            if el_type == "Title":
                chunk_texts.append(f"[Title: {el_text}]")
            elif el_type == "NarrativeText":
                chunk_texts.append(el_text)

        text_chunks.append(" ".join(chunk_texts))  

    return text_chunks

def display_base64_image(base64_code: str) -> Image:
    image_data = base64.b64decode(base64_code)
    image = Image(data=image_data)
    display(image)
    return image

if __name__ == "__main__":
    pdf_path = "example.pdf"
    api_key = "your_api_key_here" 
    chunked_data = chunk_data(pdf_path, api_key)
    print(chunked_data)