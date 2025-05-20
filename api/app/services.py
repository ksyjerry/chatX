import openai
from .config import OPENAI_API_KEY, GPT_MODEL
from .models import ChatMessage, ChatRequest, ChatResponse, ImageAnalysisRequest, ImageAnalysisResponse, WebSearchRequest, WebSearchResponse
from fastapi.responses import StreamingResponse
import json
import asyncio

# OpenAI 클라이언트 설정
client = openai.OpenAI(api_key=OPENAI_API_KEY)

async def generate_chat_response(request: ChatRequest) -> ChatResponse:
    """
    대화 응답을 생성합니다.
    
    Args:
        request: ChatRequest 모델의 요청 데이터
    
    Returns:
        ChatResponse: 응답 데이터
    """
    try:
        # 모델 설정
        model = request.model or "gpt-4.1"
        
        # 모델 ID 매핑 (필요한 경우)
        model_mapping = {
            "gpt-4.1": "gpt-4.1",
            "gpt-4o": "gpt-4.1",
            "o4-mini": "gpt-4.1",
            "o3": "gpt-3.5-turbo"
        }
        
        # 모델 ID 변환
        api_model = model_mapping.get(model, model)
        
        # 입력 메시지 형식 변환 및 필터링
        input_messages = []
        for msg in request.messages:
            # 이전 웹 검색 관련 메시지 필터링 (삼일회계법인 등 특정 키워드 포함된 메시지 제외)
            if msg.role == "system" and any(keyword in msg.content for keyword in ["삼일회계법인", "웹 검색:", "검색 결과:"]):
                print(f"Filtering out system message containing search keywords: {msg.content[:30]}...")
                continue
            
            input_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # API 호출 준비
        api_params = {
            "model": api_model,
            "messages": input_messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }
        
        # API 호출
        response = client.chat.completions.create(**api_params)
        
        # 응답 파싱
        content = response.choices[0].message.content
        
        # 사용량 정보
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return ChatResponse(
            response=content,
            model=model,
            usage=usage,
            citations=[]
        )
        
    except Exception as e:
        # 에러 처리
        error_message = f"Error generating chat response: {str(e)}"
        print(f"Chat error: {str(e)}")
        return ChatResponse(
            response=error_message,
            model=model,
            usage={"error": str(e)}
        )

async def analyze_image(request: ImageAnalysisRequest) -> ImageAnalysisResponse:
    """
    OpenAI API를 사용하여 이미지를 분석합니다.
    
    Args:
        request: ImageAnalysisRequest 모델의 요청 데이터
    
    Returns:
        ImageAnalysisResponse: 이미지 분석 결과
    """
    try:
        # 모델 설정 (기본값: GPT-4 Vision)
        model = request.model or "gpt-4.1"
        
        # 이미지 URL 확인 및 처리
        image_url = request.image_url
        
        if not image_url:
            raise ValueError("유효한 이미지 URL이 필요합니다.")
        
        # API 호출을 위한 입력 구성
        messages = []
        
        # 대화 컨텍스트가 있으면 추가
        if request.conversation_history:
            for msg in request.conversation_history:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # 사용자 메시지와 이미지 추가
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": request.prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        })
        
        # API 호출
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=request.max_tokens
        )
        
        # 응답 파싱
        content = response.choices[0].message.content
        
        # 사용량 정보
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return ImageAnalysisResponse(
            response=content,
            model=model,
            usage=usage
        )
        
    except Exception as e:
        # 에러 처리
        error_message = f"Error analyzing image: {str(e)}"
        print(f"Image analysis error: {str(e)}")
        return ImageAnalysisResponse(
            response=error_message,
            model=model or "gpt-4.1",
            usage={"error": str(e)}
        )

async def analyze_image_streaming(request: ImageAnalysisRequest):
    """
    이미지를 분석하고 스트리밍 응답을 생성합니다.
    
    Args:
        request: ImageAnalysisRequest 객체
        
    Returns:
        StreamingResponse: 스트리밍 응답 객체
    """
    # 모델 설정 (기본값: GPT-4 Vision)
    model = request.model or "gpt-4.1"
    
    async def stream_generator():
        try:
            # 이미지 URL 확인 및 처리
            image_url = request.image_url
            
            if not image_url:
                raise ValueError("유효한 이미지 URL이 필요합니다.")
            
            # API 호출을 위한 입력 구성
            messages = []
            
            # 대화 컨텍스트가 있으면 추가
            if request.conversation_history:
                for msg in request.conversation_history:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # 사용자 메시지와 이미지 추가
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": request.prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            })
            
            # API 호출
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens,
                stream=True
            )
            
            collected_messages = []
            
            # 청크 스트리밍
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    collected_messages.append(content)
                    
                    # 각 청크를 JSON으로 반환
                    yield f"data: {json.dumps({'content': content, 'is_streaming': True, 'model': model})}\n\n"
                    await asyncio.sleep(0)
            
            # 스트리밍 완료 신호
            completion_info = {
                'content': '', 
                'is_streaming': False, 
                'model': model, 
                'usage': {'completion_tokens': len(collected_messages)}
            }
            
            yield f"data: {json.dumps(completion_info)}\n\n"
            yield f"data: [DONE]\n\n"
            
        except Exception as e:
            # 에러 처리
            error_message = f"Error streaming image analysis: {str(e)}"
            print(f"Image streaming error: {str(e)}")
            yield f"data: {json.dumps({'content': error_message, 'is_streaming': False, 'error': str(e), 'model': model})}\n\n"
            yield f"data: [DONE]\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

async def generate_streaming_response(request: ChatRequest):
    """
    대화 응답을 생성하고 스트리밍 형식으로 반환합니다.
    
    Args:
        request: ChatRequest 모델의 요청 데이터
    
    Returns:
        스트리밍 응답 제너레이터
    """
    # 모델 설정
    model = request.model or "gpt-4.1"
    
    # 모델 ID 매핑 (필요한 경우)
    model_mapping = {
        "gpt-4.1": "gpt-4.1",
        "gpt-4o": "gpt-4.1",
        "o4-mini": "gpt-4.1",
        "o3": "gpt-3.5-turbo"
    }
    
    # 모델 ID 변환
    api_model = model_mapping.get(model, model)
    
    async def stream_generator():
        try:
            # 입력 메시지 형식 변환 및 필터링
            filtered_messages = []
            
            # 특수 필터링 단어 리스트
            filter_keywords = [
                "삼일회계법인", "주소는", "웹 검색:", "검색 결과:", 
                "bizbank.co.kr", "oldee.kr", "ytn.co.kr", "sedaily.com"
            ]
            
            for msg in request.messages:
                # 시스템 메시지 필터링
                if msg.role == "system" and any(keyword in msg.content for keyword in filter_keywords):
                    print(f"Filtering out system message with keywords: {msg.content[:50]}...")
                    continue
                
                # 사용자 메시지 중 웹 검색 접두사 제거
                if msg.role == "user" and msg.content.startswith("웹 검색:"):
                    msg.content = msg.content.replace("웹 검색:", "").strip()
                
                filtered_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # API 호출 준비
            api_params = {
                "model": api_model,
                "messages": filtered_messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True
            }
            
            # API 호출
            stream = client.chat.completions.create(**api_params)
            
            collected_messages = []
            
            # 청크 스트리밍
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    collected_messages.append(content)
                    
                    # 각 청크를 JSON으로 반환
                    yield f"data: {json.dumps({'content': content, 'is_streaming': True, 'model': model})}\n\n"
                    await asyncio.sleep(0)
            
            # 스트리밍 완료 신호
            completion_info = {
                'content': '', 
                'is_streaming': False, 
                'model': model, 
                'usage': {'completion_tokens': len(collected_messages)}
            }
            
            yield f"data: {json.dumps(completion_info)}\n\n"
            yield f"data: [DONE]\n\n"
            
        except Exception as e:
            # 에러 처리
            error_message = f"Error streaming response: {str(e)}"
            print(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'content': error_message, 'is_streaming': False, 'error': str(e), 'model': model})}\n\n"
            yield f"data: [DONE]\n\n"
    
    # 비동기 이터레이터 반환
    return stream_generator()

async def perform_web_search(request: WebSearchRequest) -> WebSearchResponse:
    """
    OpenAI API의 웹 검색 도구를 사용하여 웹 검색을 수행합니다.
    
    Args:
        request: WebSearchRequest 모델의 요청 데이터
    
    Returns:
        WebSearchResponse: 웹 검색 결과
    """
    try:
        # 모델 설정 - 웹 검색 지원 모델 사용
        model = "gpt-4o-search-preview"  # 웹 검색 지원 모델로 고정
        
        # API 호출 준비
        messages = [
            {
                "role": "user",
                "content": request.query
            }
        ]
        
        # 웹 검색 옵션 설정
        web_search_options = {
            "search_context_size": request.search_context_size or "medium"
        }
        
        # API 호출 - temperature 제외
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=request.max_tokens,
            web_search_options=web_search_options
        )
        
        # 응답 파싱
        content = response.choices[0].message.content
        
        # 인용 정보 추출
        citations = []
        if hasattr(response.choices[0].message, 'annotations'):
            for annotation in response.choices[0].message.annotations:
                if annotation.type == 'url_citation':
                    citations.append({
                        'url': annotation.url_citation.url,
                        'title': annotation.url_citation.title,
                        'start_index': annotation.url_citation.start_index,
                        'end_index': annotation.url_citation.end_index
                    })
        
        # 사용량 정보
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return WebSearchResponse(
            response=content,
            model=model,
            usage=usage,
            citations=citations
        )
    
    except Exception as e:
        # 에러 처리
        error_message = f"Error performing web search: {str(e)}"
        print(f"Web search error: {str(e)}")
        return WebSearchResponse(
            response=error_message,
            model=model,
            usage={"error": str(e)}
        ) 