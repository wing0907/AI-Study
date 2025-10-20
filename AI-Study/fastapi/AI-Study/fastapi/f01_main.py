from fastapi import FastAPI

app = FastAPI()         # 인스턴스화 하다

@app.get("/")           # / 경로,  http://127.0.0.1:8000 검색하면 뜸    
def read_root():
    return{"message": "Hello, World"}

# @ => decorator 는 함수를 받아들인다. read_root 를 실행하면 app.get 에 들어있는 내용도 같이 실행된다.

@app.get("/hi/")          # http://127.0.0.1:8000/hi 검색하면 뜸
def read_root():
    return{"message": "Hi, Hi, Hi"}

@app.get("/hello/")          # http://127.0.0.1:8000/hello 검색하면 뜸
def read_root():
    return{"message": "Hello, Hello, Hello"}