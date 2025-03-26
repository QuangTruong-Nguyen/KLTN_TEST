# import google.generativeai as genai
from google import genai
from typing import List
import os
from dotenv import load_dotenv


load_dotenv()

def summarize_text(text_chunks: List[str]) -> List[str]:
    '''
        Tóm tắt danh sách các đoạn văn bản bằng mô hình Gemini.
        Args:
            text_chunks (List[str]): Danh sách các đoạn văn bản cần tóm tắt.

        Returns:
            List[str]: Danh sách các đoạn văn bản đã được tóm tắt.            
    '''
    summaries = []
    gemini_key = os.getenv('GEMINI_API_KEY')
    client = genai.Client(api_key=gemini_key)

    for text in text_chunks:
        prompt = f"""
        You are a text summarization expert. You need to summarize text from a book for
        educational purposes. It is especially important to retain the titles within the text.

        The following text needs to be summarized: {text}.
        Note that you must not add any introductory sentences, such as
        'The following is the summary,...'. No need for bold or italic formatting in the text.
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
        summaries.append(response.text)

    return summaries

if __name__ == "__main__":
    example_texts = ["This is a sample text that needs to be summarized."]
    
    result = summarize_text(example_texts)
    print(result)
