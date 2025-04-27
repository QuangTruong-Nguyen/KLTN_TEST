from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List,Union, Literal, Optional



@dataclass
class Message:
    content: str
    role: str = "assistant"

@dataclass
class GraphState:
    """Mô hình trạng thái cho đồ thị."""
    question: str
    retrieval_result: str = ""
    search_result: str = ""
    outline_result: str = ""
    quizz_result: str = ""
    evaluation_result: str = ""
    next_step: str = "retrieve"

class ToolOutput(BaseModel):
    result: str

class ImageData(BaseModel):
    image_url: str
    caption: str
    width: int
    height: int
    
        
@dataclass
class Content:
    title: str
    content: str
    images: List[ImageData]
    language: str

class BulletPoints(BaseModel):
    subject: str
    points: List[str]

class Description(BaseModel):
    text: str
    
class Slide(BaseModel):
    title: str = Field(description="title of slide")
    body_text: Union[BulletPoints | Description]
    reference: str | None = Field(description="Reference link of figures, cite, etc", default=None)
    layout: Literal["cover", "table content", "only text", "text, image 25%", "text and image equal, 50%-50%",
    "image, text 25%", "only image", "text and 4 images", "text and 2 images", "graph", "video", "closing"]
    image_urls: Optional[List[str]]
    page: int    
    
class Presentation(BaseModel):
    title: str
    slides: List[Slide]    
    
class Question(BaseModel):
    question: str = Field(description="Question of multichoice")
    option: List[str] = Field(description="List of multiple")
    answer: str =Field(description="Answer of question from option, not A/B/C/D")
    explain: str = Field(description="Explain the correct answer")
    source: str =Field(description="Extract questions from database or web")
    
class OutputQuizz(BaseModel):
    questions: List[Question]