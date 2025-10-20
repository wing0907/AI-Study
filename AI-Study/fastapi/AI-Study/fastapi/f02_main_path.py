from fastapi import FastAPI

app = FastAPI()

@app.get("/")               # uvicorn f02_main_path:app --reload 
def read_root():            # dir/w   파일명 들어왔는지 확인하기
    return {"message" : "Hello, FastAPI"}  #http://127.0.0.1:8000 = http://localhost:8000/

@app.get("/item/{item_id}")        # key value 형태는 dictionary 형태
def read_item(item_id):
    return {"item_id":item_id}

@app.get("/items/")
def read_items(skip=0, limit=10):
    return {'skip': skip, "limit": limit}   # http://127.0.0.1:8000/items/?skip=5&limit=99 라고 하면 skip은 5 limit는 99로 표출된다
                                            # ctrl + c = 웹 종료   # uvicorn f02_main_path:app --reload --port=8888
 
 @app.get("pretty")
                                           