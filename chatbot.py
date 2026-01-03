from dotenv import load_dotenv
from typing import Annotated, Sequence
from typing_extensions import TypedDict
import os

# LangChain을 통한 Gemini 사용 (더 안정적)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# env 로드
load_dotenv(override=True)

# API 키 확인
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY가 설정되지 않았습니다.")

# 스테이트 정의
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 챗봇 노드 정의
def chatbot_node(state: State) -> State:
    """
    사용자 메시지를 받아 AI 응답을 생성하는 노드
    """
    # LangChain을 통한 Gemini LLM 초기화
    # 사용 가능한 모델: gemini-flash-latest, gemini-pro-latest, gemini-2.0-flash 등
    # 무료 할당량이 있는 모델 사용
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",  # 무료 티어에서 사용 가능한 모델
        temperature=0.7
    )
    
    # 메시지 가져오기
    messages = state["messages"]
    
    # LLM을 통해 응답 생성
    response = llm.invoke(messages)
    
    # 응답을 메시지에 추가
    return {"messages": [response]}


def create_chatbot_graph():
    workflow = StateGraph(State)
    workflow.add_node("chatbot", chatbot_node)
    workflow.set_entry_point("chatbot")
    workflow.add_edge("chatbot", END)
    return workflow.compile()


app = create_chatbot_graph()


if __name__ == "__main__":
    # 사용 가능한 모델 확인 (선택사항)
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        models = client.models.list()
        print("사용 가능한 모델:")
        for model in models:
            if hasattr(model, 'name'):
                print(f"  - {model.name}")
    except Exception as e:
        print(f"모델 목록 확인 실패: {e}")
    
    print("\n" + "=" * 50)
    print("LangGraph + Gemini (LangChain) 실행")
    print("=" * 50)

    inputs = {
        "messages": [
            HumanMessage(content="1 + 1 답이 뭐야?")
        ]
    }

    print(f"\n사용자: {inputs['messages'][0].content}\n")
    print("채팅봇: ", end="", flush=True)

    result = app.invoke(inputs)
    ai_message = result["messages"][-1]
    print(ai_message.content)

    print("\n" + "=" * 50)
