"""
llm_client.py
Abstraction for LLM usage with three interchangeable backends:
- ManualBackend: prints prompts to copy/paste into ChatGPT UI (zero cost).
- OpenAIBackend: (stub) calls OpenAI Agents/Responses API when you're ready.
- LocalBackend: placeholder to integrate a local model (e.g., Ollama) later.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

class ManualBackend:
    def __init__(self):
        pass

    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        print("\n=== ManualBackend ===")
        print("Copy the following into ChatGPT (GPT-5 Thinking) and paste the reply back here:")
        for m in messages:
            print(f"\n[{m.role.upper()}]\n{m.content}")
        return {"manual": True, "note": "Use ChatGPT UI to run this prompt. Paste response into pipeline if needed."}

class OpenAIBackend:
    """
    Stub: Uncomment and fill in when you want to use the API.
    Keeps your code API-ready but doesn't require a key now.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        self.api_key = api_key
        self.model = model
        if api_key is None:
            raise RuntimeError("OpenAIBackend requires an API key. Provide one via env or config.")

    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        # Example placeholder (pseudo-code):
        # from openai import OpenAI
        # client = OpenAI(api_key=self.api_key)
        # resp = client.chat.completions.create(model=self.model, messages=[m.__dict__ for m in messages])
        # return {"manual": False, "content": resp.choices[0].message.content}
        raise NotImplementedError("Wire this when you're ready to use the API.")

class LocalBackend:
    """Placeholder for a local model integration (e.g., Ollama)."""
    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        raise NotImplementedError("Implement a local model call if desired.")

class LLMClient:
    def __init__(self, backend: str = "manual", **kwargs):
        if backend == "manual":
            self.backend = ManualBackend()
        elif backend == "openai":
            self.backend = OpenAIBackend(**kwargs)
        elif backend == "local":
            self.backend = LocalBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return self.backend.generate(messages, tools)
