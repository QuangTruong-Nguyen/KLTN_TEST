from __future__ import annotations
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext, End, Graph
from models import GraphState, ToolOutput
from agents import *  
from tools import *
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from IPython.display import Image, display
import asyncio
import yaml

def load_prompts_from_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompts_data = yaml.safe_load(file)
    return prompts_data

prompts = load_prompts_from_yaml('D:/KLTN2/pydanticAI/prompt.yaml')


quizz_agent, outline_agent, evaluate_agent=initialize_agents()
@dataclass
class RetrieveNode(BaseNode[GraphState]):
    """Node thực hiện truy xuất dữ liệu từ vector store."""
    async def run(self, ctx: GraphRunContext[GraphState]) -> SearchNode:
        result = retrieve(ctx.state.question)
        print(f"Kết quả truy xuất: {result}")
        ctx.state.retrieval_result = result
        ctx.state.next_step = "search"
        return SearchNode()

@dataclass
class SearchNode(BaseNode[GraphState]):
    """Node thực hiện tìm kiếm trên web."""
    async def run(self, ctx: GraphRunContext[GraphState]) -> OutlineNode:
        result = tavily_search(ctx.state.question)
        print(f"Kết quả tìm kiếm web: {result}")
        ctx.state.search_result = result
        ctx.state.next_step = "outline"
        return OutlineNode()

@dataclass
class OutlineNode(BaseNode[GraphState]):
    """Node tạo dàn ý học tập."""
    async def run(self, ctx: GraphRunContext[GraphState]) -> QuizzNode:
        
        print("_______OUTLINE______________")
        prompt=prompts['outlineNode'].format(
            retrieval_result=ctx.state.retrieval_result,
            search_result=ctx.state.search_result            
        )
        result = await outline_agent.run(prompt)
        print(f"Dàn ý học tập: {result.data}")
        ctx.state.outline_result = result.data
        ctx.state.next_step = "quizz"
        return QuizzNode()

@dataclass
class QuizzNode(BaseNode[GraphState]):
    """Node tạo câu hỏi trắc nghiệm."""
    async def run(self, ctx: GraphRunContext[GraphState]) -> EvaluateNode:      
        print("Quizz")
        prompt=prompts['quizzNode'].format(
            retrieval_result=ctx.state.retrieval_result,
            search_result=ctx.state.search_result 
        )
        
        result = await quizz_agent.run(prompt)
        print(f"Câu hỏi trắc nghiệm: {result.data}")
        ctx.state.quizz_result = result.data
        ctx.state.next_step = 'evaluate' 
        return EvaluateNode()

@dataclass
class EvaluateNode(BaseNode[GraphState]):
    """Node đánh giá kết quả."""
    async def run(self, ctx: GraphRunContext[GraphState]) -> SupervisorNode:
        print("____Evaluate____")
        prompt = f"""
            Bộ câu hỏi : {ctx.state.quizz_result}
            """
        result =await evaluate_agent.run(prompt)
        print(f"Đánh giá kết quả: {result.data}")
        ctx.state.evaluate_result = result.data
        ctx.state.next_step = "end"
        return SupervisorNode()

@dataclass
class SupervisorNode(BaseNode[GraphState]):
    """Supervisor quyết định bước tiếp theo dựa trên trạng thái hiện tại."""
    async def run(self, ctx: GraphRunContext[GraphState]) -> RetrieveNode | SearchNode | OutlineNode | QuizzNode | EvaluateNode | End[dict]:
        print(f"Supervisor đang xử lý. Bước tiếp theo: {ctx.state.next_step}")

        next_step = ctx.state.next_step
        if next_step == "retrieve":
            return RetrieveNode()
        elif next_step == "search":
            return SearchNode()
        elif next_step == "outline":
            return OutlineNode()
        elif next_step == "quizz":
            return QuizzNode()
        elif next_step == 'evaluate':
            return EvaluateNode()
        else:
            final_result = {
                "question": ctx.state.question,
                "retrieval_result": ctx.state.retrieval_result,
                "search_result": ctx.state.search_result,
                "outline_result": ctx.state.outline_result,
                "evaluate_result": ctx.state.evaluate_result,
                "quizz_result": ctx.state.quizz_result
            }
            return End(final_result)


async def main():
    """Main function to run the graph."""
    initial_question = "Tôi muốn biết về Basics Data Types"
    initial_state = GraphState(question=initial_question)

    workflow = Graph(
        nodes=[RetrieveNode, SearchNode, OutlineNode, QuizzNode, EvaluateNode, SupervisorNode]
    )

    result = await workflow.run(SupervisorNode(), state=initial_state)

    print("\n--- KẾT QUẢ CUỐI CÙNG ---")
    print(f"Câu hỏi: {initial_question}")
    print(f"Kết quả truy xuất: {result.output['retrieval_result']}")
    print(f"Kết quả tìm kiếm: {result.output['search_result']}")
    print(f"Dàn ý học tập: {result.output['outline_result']}")
    print(f"Câu hỏi trắc nghiệm: {result.output['quizz_result']}")


    texxt=f'''
    Kết quả truy xuất: {result.output['retrieval_result']}
    Kết quả tìm kiếm: {result.output['search_result']}
    Dàn ý học tập: {result.output['outline_result']}
    Câu hỏi trắc nghiệm: {result.output['quizz_result']}
    '''

    with open("output.txt",'w',encoding='utf-8')  as f:
        f.write(texxt)
    # Image(workflow.mermaid_image(start_node=SupervisorNode))
    with open('output_image.png', 'wb') as f:
        f.write(workflow.mermaid_image(start_node=SupervisorNode))
    return result.output

if __name__ == "__main__":
    asyncio.run(main())
