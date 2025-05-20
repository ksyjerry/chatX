from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import router
from .config import HOST, PORT

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="GPT-4.1 API",
    description="OpenAI GPT-4.1 모델을 사용한 채팅 API",
    version="1.0.0"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 프로덕션에서는 출처를 명시적으로 지정해야 합니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(router, prefix="/api")

# 기본 경로
@app.get("/")
async def root():
    return {
        "message": "GPT-4.1 API 서버가 실행 중입니다.",
        "docs_url": "/docs",
        "endpoints": {
            "chat": "/api/chat",
            "models": "/api/models"
        }
    } 