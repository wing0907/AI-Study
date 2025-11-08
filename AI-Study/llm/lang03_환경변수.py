# 환경 변수 방법
#################################
"""
1. 시작 - 환경치면 - 시스템 환경 변수 편집 - 사용자 환경 변수 편집
2. 새로 만들기 - 변수 이름: OPENAI_API_KEY, 변수 값: sk-xxxxx
3. 재부팅 또는 터미널 재시작
4. python-dotenv 설치: pip install python-dotenv
5. .env 파일 생성 - OPENAI_API_KEY=sk-xxxxx
"""
#################################

from openai import OpenAI
from dotenv import load_dotenv

# load_dotenv()  # .env 파일에서 환경 변수 로드 / 환경변수 키 호출

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "temperature": 0.9,
        "content": "Hello!"
        }
    ]
)    

print(completion)
print('='*80)
print(completion.choices[0].message.content)


