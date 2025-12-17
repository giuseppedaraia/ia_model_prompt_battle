
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from anthropic import Anthropic
import google.generativeai as genai
from mistralai import Mistral

app = FastAPI()

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    model_provider: str
    model_name: str
    api_key: str

class GenerateResponse(BaseModel):
    provider: str
    model: str
    content: str
    error: Optional[str] = None

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        if request.model_provider == "openai":
            client = openai.OpenAI(api_key=request.api_key)
            response = client.chat.completions.create(
                model=request.model_name,
                messages=[{"role": "user", "content": request.prompt}]
            )
            return GenerateResponse(
                provider="openai",
                model=request.model_name,
                content=response.choices[0].message.content
            )

        elif request.model_provider == "anthropic":
            client = Anthropic(api_key=request.api_key)
            response = client.messages.create(
                model=request.model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": request.prompt}]
            )
            return GenerateResponse(
                provider="anthropic",
                model=request.model_name,
                content=response.content[0].text
            )

        elif request.model_provider == "google":
            genai.configure(api_key=request.api_key)
            model = genai.GenerativeModel(request.model_name)
            response = model.generate_content(request.prompt)
            return GenerateResponse(
                provider="google",
                model=request.model_name,
                content=response.text
            )

        elif request.model_provider == "mistral":
            # Mistral's SDK might differ slightly, checking generic usage
            client = Mistral(api_key=request.api_key)
            response = client.chat.complete(
                model=request.model_name,
                messages=[{"role": "user", "content": request.prompt}]
            )
            return GenerateResponse(
                provider="mistral",
                model=request.model_name,
                content=response.choices[0].message.content
            )

        elif request.model_provider == "deepseek":
             # DeepSeek is OpenAI compatible
            client = openai.OpenAI(
                api_key=request.api_key,
                base_url="https://api.deepseek.com"
            )
            response = client.chat.completions.create(
                model=request.model_name,
                messages=[{"role": "user", "content": request.prompt}]
            )
            return GenerateResponse(
                provider="deepseek",
                model=request.model_name,
                content=response.choices[0].message.content
            )

        elif request.model_provider == "grok":
            # Grok (xAI) is OpenAI compatible
            client = openai.OpenAI(
                api_key=request.api_key,
                base_url="https://api.x.ai/v1"
            )
            response = client.chat.completions.create(
                model=request.model_name,
                messages=[{"role": "user", "content": request.prompt}]
            )
            return GenerateResponse(
                provider="grok",
                model=request.model_name,
                content=response.choices[0].message.content
            )

        else:
            raise HTTPException(status_code=400, detail="Unknown provider")

    except Exception as e:
        return GenerateResponse(
            provider=request.model_provider,
            model=request.model_name,
            content="",
            error=str(e)
        )

# Serve static files for frontend
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
