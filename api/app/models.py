from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any, Literal


class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class ImageAnalysisRequest(BaseModel):
    """
    이미지 분석 요청 모델
    """
    image_url: Optional[str] = None
    prompt: str
    model: Optional[str] = None
    max_tokens: int = 1000
    detail: str = "auto"
    stream: bool = False
    conversation_history: Optional[List[ChatMessage]] = None


class ImageAnalysisResponse(BaseModel):
    response: str
    model: str
    usage: dict


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False
    enable_web_search: Optional[bool] = False
    search_query: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    model: str
    usage: dict
    is_streaming: Optional[bool] = False
    citations: Optional[List[Dict[str, str]]] = None


class WebSearchRequest(BaseModel):
    query: str
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    search_context_size: Optional[str] = "medium"  # low, medium, high 중 하나
    # 예제 user_location:
    # {
    #   "country": "KR",
    #   "city": "Seoul",
    #   "region": "Seoul",
    #   "timezone": "Asia/Seoul"
    # }
    user_location: Optional[Dict[str, str]] = None


class WebSearchResponse(BaseModel):
    response: str
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    citations: Optional[List[Dict[str, Union[str, int]]]] = None 