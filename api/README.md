# GPT-4.1 FastAPI 서버

OpenAI의 GPT-4.1 모델을 사용하여 채팅 응답을 생성하는 FastAPI 서버입니다.

## 설치 방법

1. 의존성 설치:

   ```bash
   pip install -r requirements.txt
   ```

2. 환경 변수 설정:
   `.env` 파일을 생성하고 다음 변수를 설정합니다:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   HOST=0.0.0.0
   PORT=8000
   GPT_MODEL=gpt-4-1106-preview
   ```

## 실행 방법

```bash
python run.py
```

서버는 기본적으로 `http://localhost:8000`에서 실행됩니다.

## API 엔드포인트

### 채팅 API

- URL: `/api/chat`
- 메서드: `POST`
- 설명: 채팅 메시지를 처리하고 GPT 응답을 반환합니다.
- 요청 예시:
  ```json
  {
    "messages": [
      {
        "role": "user",
        "content": "안녕하세요, GPT-4.1!"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 1000
  }
  ```

### 모델 목록 API

- URL: `/api/models`
- 메서드: `GET`
- 설명: 사용 가능한 모델 목록을 반환합니다.

## API 문서

API 문서는 `/docs` 또는 `/redoc`에서 확인할 수 있습니다.
