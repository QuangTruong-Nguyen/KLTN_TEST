from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
# from pydantic_ai.models.cohere import CohereModel
from models import ToolOutput
import yaml
import os
from dotenv import load_dotenv
load_dotenv()


def load_prompts_from_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompts_data = yaml.safe_load(file)
    return prompts_data
    
def initialize_llm_models():
    """Initialize language models."""
    llm = GroqModel(
        'llama-3.3-70b-versatile', 
        provider=GroqProvider(api_key=os.getenv('API_KEY_GROQ'))
    )
    
    llm2 = GeminiModel(
        'gemini-2.0-flash', 
        provider=GoogleGLAProvider(api_key=os.getenv('API_GEMINI_MODEL'))
    )
    
    return llm, llm2

def initialize_agents():
    """Initialize agents with the provided prompts and language models."""
    
    prompts = load_prompts_from_yaml('./pydanticAI/prompt.yaml')
    llm, llm2 = initialize_llm_models()
    
    quizz_agent = Agent(
        llm,
        result_type=ToolOutput,
        system_prompt=prompts['prompt_quizz']
    )
    
    outline_agent = Agent(
        llm2,
        result_type=ToolOutput,
        system_prompt=prompts['prompt_outline']
    )
    
    evaluate_agent = Agent(
        llm,
        result_type=ToolOutput,
        system_prompt=prompts['prompt_evaluate']
    )
    
    return quizz_agent, outline_agent, evaluate_agent

if __name__ == "__main__":

    file_path = './pydanticAI/prompt.yaml'
    
    # Load prompts
    prompts = load_prompts_from_yaml(file_path)
    
    # print(prompts['prompt_evaluate'])
    # Initialize language models
    # llm, llm2 = initialize_llm_models()
    
    # Initialize agents
    # quizz_agent, outline_agent, evaluate_agent = initialize_agents(prompts, llm, llm2)
    