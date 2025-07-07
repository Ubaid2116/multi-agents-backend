# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import asyncio
from dotenv import load_dotenv
from collections import defaultdict
from agents import Agent, Runner
from agents import ModelSettings, RunConfig, OpenAIChatCompletionsModel, AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")
if not google_api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

external_client = AsyncOpenAI(
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

content_agent = Agent(
    name="ContentWriter",
    instructions="You’re a specialist content writer. Handle only content creation tasks."
)
marketing_agent = Agent(
    name="DigitalMarketer",
    instructions="Provide marketing strategy, SEO tips, copywriting."
)
webdev_agent = Agent(
    name="WebDeveloper",
    instructions="Answer only web development questions: code, deployment, debugging."
)
manager_agent = Agent(
    name="Manager",
    instructions=(
        "You’re the manager. Route queries: web dev → WebDeveloper; "
        "content → ContentWriter; marketing → DigitalMarketer. "
        "Handle greetings and identity as specified."
    ),
    handoffs=[content_agent, marketing_agent, webdev_agent]
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)
run_config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://multi-agents-openai.vercel.app", "https://kzmihj4sv8yv1gxiaw6g.lite.vusercontent.net/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
session_memories = defaultdict(list)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, request: Request):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    session_id = request.headers.get("X-Session-Id", "default")
    memory = session_memories[session_id]
    memory.append({"role": "user", "content": req.message})
    result = await Runner.run(manager_agent, req.message, run_config=run_config, context={"memory": memory})
    reply = result.final_output
    memory.append({"role": "assistant", "content": reply})
    return {"reply": reply}

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    session_id = request.headers.get("X-Session-Id", "default")
    memory = session_memories[session_id]
    memory.append({"role": "user", "content": req.message})
    result = Runner.run_streamed(manager_agent, req.message, run_config=run_config, context={"memory": memory})

    async def event_generator():
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta
    return StreamingResponse(event_generator(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
