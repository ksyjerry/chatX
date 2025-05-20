import requests
import json

# 서버 URL
BASE_URL = "http://localhost:8000"

def test_chat_api():
    """
    채팅 API 테스트
    """
    url = f"{BASE_URL}/api/chat"
    
    # 요청 데이터
    data = {
        "messages": [
            {
                "role": "user",
                "content": "안녕하세요, GPT-4.1! 자기소개 부탁드려요."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    # API 호출
    response = requests.post(url, json=data)
    
    # 응답 확인
    if response.status_code == 200:
        result = response.json()
        print("=== GPT 응답 ===")
        print(result["response"])
        print("\n=== 사용 정보 ===")
        print(f"모델: {result['model']}")
        print(f"토큰 사용량: {result['usage']}")
    else:
        print(f"에러 발생: {response.status_code}")
        print(response.text)

def test_models_api():
    """
    모델 목록 API 테스트
    """
    url = f"{BASE_URL}/api/models"
    
    # API 호출
    response = requests.get(url)
    
    # 응답 확인
    if response.status_code == 200:
        result = response.json()
        print("=== 사용 가능한 모델 ===")
        for model in result["models"]:
            print(f"{model['id']} - {model['name']}")
    else:
        print(f"에러 발생: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("API 테스트 시작...")
    print("\n1. 모델 목록 테스트")
    test_models_api()
    
    print("\n2. 채팅 API 테스트")
    test_chat_api()
    
    print("\n테스트 완료!") 