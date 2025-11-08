from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일에서 환경 변수 로드 / 환경변수 키 호출

client = OpenAI()

while True:
    user_input = input("나 : ")
    
    if user_input == "exit":
        print("대화를 종료합니다.")
        break
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.9,
        messages=[
            {"role": "system", "content":"너는 나를 서포트하는 AI다"},
            {"role": "user", "content": user_input}
        ]
    )
    print("AI : ", response.choices[0].message.content)

# 실질적으로 04_멀티턴1은 싱글턴이다. 멀티턴을 설명하기 위한 사전 작업이다
# 멀티턴은 재학습이 아니다. 사용자가 입력한 내용을 기억하는 것이다.
# 멀티턴은 사용자가 입력한 내용을 messages에 추가하는 것이다.




