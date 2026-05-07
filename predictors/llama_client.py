
import requests
import json
from typing import Any, List, Optional, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

class LocalLlamaClient(BaseChatModel):
    base_url: str = "http://127.0.0.1:8088/v1"
    model_name: str = "local-model"
    temperature: float = 0.0
    max_tokens: int = 4096
    stop: List[str] = []

    @property
    def _llm_type(self) -> str:
        return "local-llama-custom"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            
            content = msg.content
            # Handle list content (multimodal)
            if isinstance(content, list):
                # Ensure structure matches OpenAI vision
                new_content = []
                for item in content:
                    if isinstance(item, dict):
                         # If it has 'image_url', keep it
                         if "image_url" in item:
                             new_content.append(item)
                         elif "text" in item:
                             new_content.append(item)
                         elif "type" in item and item["type"] == "text":
                             new_content.append(item)
                         elif "type" in item and item["type"] == "image_url":
                             new_content.append(item)
                    elif isinstance(item, str):
                        new_content.append({"type": "text", "text": item})
                content = new_content

            openai_messages.append({"role": role, "content": content})

        payload = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        # Merge stop tokens
        final_stop = stop or self.stop
        if final_stop:
            payload["stop"] = final_stop
            
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
            choice = result["choices"][0]
            message_data = choice["message"]
            content = message_data.get("content", "")
            reasoning = message_data.get("reasoning_content", "")
            
            # Critical Fix: Preserve reasoning
            additional_kwargs = {"reasoning_content": reasoning}
            
            # Logging fix
            if not content and reasoning:
                print(f"[LocalLlamaClient] ⚠️ Content empty, found reasoning ({len(reasoning)} chars). Using reasoning as content.")
                # We can either set content=reasoning OR keep it separate.
                # For compatibility with downstream parsers expecting text, let's use reasoning as content if content is empty.
                # BUT, let's prepend it?
                # The user wants "thought process as well".
                # If we put it in content, the user sees it.
                # Let's put it in content!
                content = reasoning
                # Also keep in additional_kwargs for debugging
            
            generation = ChatGeneration(
                message=AIMessage(content=content, additional_kwargs=additional_kwargs),
                generation_info={"finish_reason": choice.get("finish_reason"), "raw_response": result}
            )
            return ChatResult(generations=[generation])
            
        except Exception as e:
            print(f"[LocalLlamaClient] Error: {e}")
            raise e
