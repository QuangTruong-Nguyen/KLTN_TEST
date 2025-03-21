from dataclasses import dataclass
from pydantic import BaseModel

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