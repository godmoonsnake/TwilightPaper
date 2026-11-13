import json
from collections.abc import Sequence
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.tool import default_tool_parser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import SecretStr


class APIRequestError(RuntimeError):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"Model API returned HTTP {status_code}.")


class APIChatModel(BaseChatModel):
    model: str
    base_url: str
    api_key: SecretStr

    @property
    def _llm_type(self) -> str:
        return "openai-compatible-http"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key.get_secret_value()}"},
            json={
                "model": self.model,
                "messages": [self._encode_message(message) for message in messages],
                **({"stop": stop} if stop else {}),
                **kwargs,
            },
            timeout=90,
        )
        if response.is_error:
            raise APIRequestError(response.status_code)
        message = response.json()["choices"][0]["message"]
        calls, invalid_calls = default_tool_parser(message.get("tool_calls") or [])
        answer = AIMessage(
            content=message.get("content") or message.get("refusal") or "",
            tool_calls=calls,
            invalid_tool_calls=invalid_calls,
        )
        return ChatResult(generations=[ChatGeneration(message=answer)])

    @staticmethod
    def _encode_message(message: BaseMessage) -> dict:
        role = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}
        payload = {"role": role[message.type], "content": message.content}
        if isinstance(message, ToolMessage):
            payload["tool_call_id"] = message.tool_call_id
        if isinstance(message, AIMessage) and message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
                }
                for call in message.tool_calls
            ]
        return payload
