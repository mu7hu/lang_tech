"""
Local inference via Ollama (or compatible API). FR-5.x.
"""

import time
from pathlib import Path

from groq import Groq
import yaml
import requests

# client = Groq()

client = Groq()

def load_models_config() -> dict:
    config_dir = Path(__file__).resolve().parent.parent / "config"
    with open(config_dir / "models.yaml") as f:
        return yaml.safe_load(f)

def get_client_config() -> dict:
    return {}

def complete(
    prompt: str,
    model: str,
    system_prompt: str | None = None,
    base_url: str | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    **kwargs # Accept extra arguments from main.py, but ignore them for Groq
) -> str:
    
    """
    Call Groq API. Returns the generated text.
    """
    
    # Optional: If your models.yaml has different names than what Groq expects,
    # you can map them here. If they match exactly, this isn't needed.
    model_name = model
    
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=0.0,
            max_tokens=256 # Only need the answer letter
        )
        return chat_completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Groq API Error: {e}")
        return ""

