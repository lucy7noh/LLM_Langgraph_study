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


# FastAPI 설정
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

# FastAPI 앱 생성
api_app = FastAPI(
    title="Proovy Chatbot API",
    description="LangGraph + Gemini 기반 챗봇 API",
    version="1.0.0"
)

# CORS 설정 (프론트엔드 연동을 위해)
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 모델
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # 대화 세션 관리용

class ChatResponse(BaseModel):
    response: str
    session_id: str

# 세션별 대화 상태 저장 (실제로는 DB 사용 권장)
chat_sessions = {}

@api_app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "status": "running",
        "service": "Proovy Chatbot API",
        "version": "1.0.0"
    }

@api_app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    챗봇과 대화하는 엔드포인트
    
    - message: 사용자 메시지
    - session_id: 대화 세션 ID (없으면 새로 생성)
    """
    try:
        # 세션 관리
        session_id = request.session_id or f"session_{len(chat_sessions)}"
        
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {"messages": []}
        
        # 사용자 메시지를 HumanMessage로 변환
        user_message = HumanMessage(content=request.message)
        chat_sessions[session_id]["messages"].append(user_message)
        
        # LangGraph 실행
        result = app.invoke({
            "messages": chat_sessions[session_id]["messages"]
        })
        
        # AI 응답 추출
        ai_message = result["messages"][-1]
        response_text = ai_message.content
        
        # 세션 상태 업데이트 (대화 히스토리 유지)
        chat_sessions[session_id] = result
        
        return ChatResponse(
            response=response_text,
            session_id=session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"챗봇 처리 중 오류 발생: {str(e)}")

@api_app.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    """대화 세션 초기화"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return {"message": "세션이 삭제되었습니다."}
    return {"message": "세션을 찾을 수 없습니다."}

@api_app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}

if __name__ == "__main__":
    # 서버 실행
    # 방법 1: reload 없이 실행 (경고 없음)
    uvicorn.run(
        api_app,
        host="0.0.0.0",
        port=8000
    )
    
    # 방법 2: reload 기능을 사용하려면 터미널에서 실행:
    # uvicorn chatbot:api_app --reload --host 0.0.0.0 --port 8000
