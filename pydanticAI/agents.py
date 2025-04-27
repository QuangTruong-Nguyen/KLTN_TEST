from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
# from pydantic_ai.models.cohere import CohereModel
from models import ToolOutput, Content, Presentation,OutputQuizz
import yaml
import os
from dotenv import load_dotenv
load_dotenv()


from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

llm = ChatGroq(temperature=0.5, model_name="llama3-70b-8192",
                api_key="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")

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
        result_type=OutputQuizz,
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

def slide_agent():
    
    slide_gen_agent = Agent(
    'google-gla:gemini-2.0-flash',
    deps_type=Content,
    result_type=Presentation,
    model_settings={"temperature": 0.0}
    )
    
    @slide_gen_agent.system_prompt
    def system_prompt(ctx: RunContext[Content]) -> str:
        return f"""
        You are assistant agent to help me generate slides for presentation.
        Use provided information {ctx.deps.content} about {ctx.deps.title} to summarize and synthesize content for the presentation.
        Given the following images information. Given each slide have size 1920x1080 px:
        IMAGES:
        """ + "\n".join(
            [f"{image.image_url}: {image.caption} have size {image.width}x{image.height}" for image in ctx.deps.images])
        + """
        Firstly, use given information to summarize and synthesize content for the presentation.
        The first slide is COVER, only include title of the presentation and subtitle. 
        The last slide is CLOSING, only include "Thanks for watching".

        Secondly, select approriate image for the slide. If have not the suitable image for the slide, simply return `None`.
        Finally, select the most appropriate layout for this slide, based on whether slide have image and its size.


        MUST USE `only text` for slide which `image_url` is empty.
        USE `only image` for slide which have large image.
        Make sure content brevity BUT clarity and meaningful.
        
        Create from 10-20 slide.
        
        Return as Example:
        Attention is All You Need
        Presenter: Alex Chandler
        09-13-2022
        ---------------------
        Motivation
        ● Transformers were developed to solve the problem machine
        translation and sequence transduction, or neural machine
        translation … but they do so much more!
        ○ Great performance in computer vision like ViT
        (Dosovitskiy, 2020)
        ○ Image Classification (CoCa Transformer)
        ○ Semantic Segmentation (ex: FD-SwinV2-G
        Transformer)
        ○ Object Detection (ex: FD-SwinV2-G Transformer)
        ---------------------
        Extended Readings
        Good Sources for Explainability:
        ● Blogs:
        ○ https://jinglescode.github.io/2020/05/27/illustrated-gui
        de-transformer/
        ○ https://towardsdatascience.com/transformers-141e32
        e69591
        ○ https://jalammar.github.io/illustrated-transformer/
        ○ https://medium.com/analytics-vidhya/neural-machine-t
        ranslation-using-bahdanau-attention-mechanism-d496
        c9be30c3
        ● Youtube:
        ○ https://www.youtube.com/watch?v=TQQlZhbC5ps
        Relevant Papers:
        ● Attention is All You Need:
        https://arxiv.org/abs/1706.03762
        ● BERT: https://arxiv.org/abs/1810.04805
        ● Roberta: https://arxiv.org/abs/1907.11692
        ● Swin Transformer: Hierarchical Vision
        Transformer using Shifted Windows:
        https://arxiv.org/abs/2103.14030
        ------------------
        Summary
        ● Transformers:
        ○ Dominate Everything
        ○ State of the art in Image Classification,
        Language Modeling, Speech Recognition,
        Vision Transformers (ex: ViT), Semantic
        Segmentation, Behavior / Decision Making,
        etc.
        """
    
    return slide_gen_agent

if __name__ == "__main__":

    file_path = './pydanticAI/prompt.yaml'
    
    # Load prompts
    prompts = load_prompts_from_yaml(file_path)
    
    # print(prompts['prompt_evaluate'])
    # Initialize language models
    # llm, llm2 = initialize_llm_models()
    
    # Initialize agents
    # quizz_agent, outline_agent, evaluate_agent = initialize_agents(prompts, llm, llm2)
    