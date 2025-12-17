
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>AI Prompt Tester</title>" in response.text

@patch("openai.resources.chat.completions.Completions.create")
def test_openai_generate(mock_create):
    # Mock response structure
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Mocked OpenAI response"))
    ]
    mock_create.return_value = mock_response

    response = client.post(
        "/api/generate",
        json={
            "prompt": "Hello",
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "api_key": "sk-fake"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "openai"
    assert data["content"] == "Mocked OpenAI response"
    assert data["error"] is None

@patch("anthropic.resources.messages.Messages.create")
def test_anthropic_generate(mock_create):
    # Mock response structure
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Mocked Claude response")]
    mock_create.return_value = mock_response

    response = client.post(
        "/api/generate",
        json={
            "prompt": "Hello",
            "model_provider": "anthropic",
            "model_name": "claude-3-sonnet",
            "api_key": "sk-ant-fake"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "anthropic"
    assert data["content"] == "Mocked Claude response"

@patch("google.generativeai.GenerativeModel.generate_content")
def test_google_generate(mock_generate):
    # Mock response structure
    mock_response = MagicMock()
    mock_response.text = "Mocked Gemini response"
    mock_generate.return_value = mock_response

    response = client.post(
        "/api/generate",
        json={
            "prompt": "Hello",
            "model_provider": "google",
            "model_name": "gemini-pro",
            "api_key": "fake-key"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "google"
    assert data["content"] == "Mocked Gemini response"

def test_unknown_provider():
    response = client.post(
        "/api/generate",
        json={
            "prompt": "Hello",
            "model_provider": "unknown_provider",
            "model_name": "whatever",
            "api_key": "fake-key"
        }
    )
    assert response.status_code == 200 # The app catches exceptions and returns error field
    data = response.json()
    assert data["error"] is not None
    assert "Unknown provider" in data["error"]
