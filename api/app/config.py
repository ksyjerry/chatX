import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# 서버 설정
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# GPT 모델 설정
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4-1106-preview") 