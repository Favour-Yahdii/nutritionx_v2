
from fastapi import FastAPI
from function_call2 import ModelInterface
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


class Input(BaseModel):
    input_text: str

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_interface = ModelInterface()

@app.post("/chat_messages/")
def chat_messages(input: Input):

    agent_response = ModelInterface.handle_function_call(user_input=input.input_text)
    print(agent_response)
    return {"agent": agent_response["response"]}
    
@app.get("/status/")
def status():
    return{"status": "OK"}
