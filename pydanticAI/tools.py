from models import ToolOutput
from tavily import TavilyClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
load_dotenv()


def embedding():
    embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')
    
    return embeddings

def retrieve(question: str) -> ToolOutput:
    """Retrieve documents from vector store."""
    print("---RETRIEVE---")
    
    embedings=embedding()
    
    vector_store = Chroma(
        collection_name="example_collection2",
        embedding_function=embedings,  # Phải sử dụng cùng embedding function
        persist_directory="./chroma_langchain_db"
    )
    try:
        # Retrieval
        documents = vector_store.similarity_search(
            question,
            k=2
        )
        result = "Result Retrieval: " + documents[0].page_content + documents[1].page_content
        return ToolOutput(result=result)
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return ToolOutput(result=f"Simulated retrieval result for question: {question}")

def tavily_search(question: str) -> ToolOutput:
    """Search the web using Tavily."""
    print("---Tavily---")
    try:
        # Tool logic here
        client = TavilyClient(os.getenv('API_TAVILY'))
        response = client.search(
            query=question,
            max_results=1
        )
        return ToolOutput(result="Research Web by Tavily: " + response['results'][0]['content'])
    except Exception as e:
        print(f"Error getting related web content: {e}")
        return ToolOutput(result=f"Simulated web search result for question: {question}")
    
    

if __name__ == "__main__":
    # result=retrieve("Khai thác dữ liệu ?")
    result=tavily_search("Data Mining")
    print(result)