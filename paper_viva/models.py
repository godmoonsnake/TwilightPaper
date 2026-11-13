import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values
from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_MODELS = {
    "ollama": "qwen3:4b",
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-5.4-mini",
    "groq": "llama-3.3-70b-versatile",
}
PROVIDER_LABELS = {"ollama": "Ollama", "gemini": "Google Gemini", "openai": "OpenAI", "groq": "Groq"}
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def config_value(name: str) -> str:
    return (os.getenv(name) or dotenv_values(ENV_FILE).get(name) or "").strip()


class ModelConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    name: str

    @classmethod
    def from_env(cls, provider: str | None = None) -> "ModelSettings":
        configured = (config_value("PAPER_MODEL_PROVIDER") or "ollama").lower()
        provider = provider or configured
        if provider not in DEFAULT_MODELS:
            raise ModelConfigurationError(
                "Set PAPER_MODEL_PROVIDER to ollama, gemini, groq, or openai in .env."
            )
        name = (config_value("PAPER_MODEL") if provider == configured else "") or DEFAULT_MODELS[provider]
        return cls(provider, name)

    def connect(self) -> BaseChatModel:
        if self.provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=self.name,
                base_url=config_value("OLLAMA_BASE_URL") or "http://127.0.0.1:11434",
                temperature=0, reasoning=False, num_ctx=8192,
                client_kwargs={"timeout": 120},
            )

        if self.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            key = config_value("GOOGLE_API_KEY") or config_value("GEMINI_API_KEY")
            if not key:
                raise ModelConfigurationError("Add GOOGLE_API_KEY to the project's .env file.")
            return ChatGoogleGenerativeAI(
                model=self.name, api_key=key, temperature=0, timeout=90, max_retries=1
            )

        if self.provider in {"openai", "groq"}:
            from .api_chat import APIChatModel

            key_name = "OPENAI_API_KEY" if self.provider == "openai" else "GROQ_API_KEY"
            key = config_value(key_name)
            if not key:
                raise ModelConfigurationError(f"Add {key_name} to the project's .env file.")
            return APIChatModel(
                model=self.name, api_key=key,
                base_url="https://api.openai.com/v1" if self.provider == "openai"
                else "https://api.groq.com/openai/v1",
            )

        raise ModelConfigurationError("Choose a supported API provider.")
