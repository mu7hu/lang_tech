"""
Local inference via Ollama (or compatible API). FR-5.x.
"""

import time
from pathlib import Path

from groq import Groq
import yaml
import requests

client = Groq()

def load_models_config() -> dict:
    config_dir = Path(__file__).resolve().parent.parent / "config"
    with open(config_dir / "models.yaml") as f:
        return yaml.safe_load(f)

def get_client_config() -> dict:
    return {}
# def get_client_config() -> dict:
#     cfg = load_models_config()
#     inf = cfg.get("inference", {}) or {}
#     return {
#         "base_url": inf.get("base_url", "http://localhost:11434"),
#         "timeout": inf.get("timeout_seconds", 120),
#         "max_retries": inf.get("max_retries", 2),
#     }


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
            max_tokens=5 # Only need the answer letter
        )
        return chat_completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Groq API Error: {e}")
        return ""
    # """
    # Call Ollama generate API. Returns the generated text.
    # On timeout or non-200, raises or returns empty string after retries (caller can treat as invalid).
    # """
    # client = get_client_config()
    # url = (base_url or client["base_url"]).rstrip("/") + "/api/generate"
    # timeout = timeout if timeout is not None else client["timeout"]
    # max_retries = max_retries if max_retries is not None else client["max_retries"]

    # payload = {
    #     "model": model,
    #     "prompt": prompt,
    #     "stream": False,
    # }
    # if system_prompt:
    #     payload["system"] = system_prompt

    # last_error = None
    # for attempt in range(max_retries + 1):
    #     try:
    #         r = requests.post(url, json=payload, timeout=timeout)
    #         r.raise_for_status()
    #         data = r.json()
    #         return data.get("response", "").strip()
    #     except requests.exceptions.Timeout as e:
    #         last_error = e
    #     except requests.exceptions.RequestException as e:
    #         last_error = e
    #     if attempt < max_retries:
    #         time.sleep(2 ** attempt)
    # if last_error:
    #     raise last_error
    # return ""
