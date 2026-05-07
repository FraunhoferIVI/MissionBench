from email.mime import image
from pathlib import Path
import pathlib
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from abc import ABC, abstractmethod
from typing import Union
from PIL import Image
import io
import os
from prompter.prompt_generator import generate_prompt
MAX_B64_BYTES = 4_800_000  # safe margin under Bedrock's 5MB limit

class VisionChain(ABC):
    def __init__(self, mission: dict = None):
        self.model_name = None  # Set this to a valid model string before use (e.g., "gpt-4o", "claude-3-sonnet-20240229")
        self.mission = mission
        self.verbose = mission["verbose"] if mission and "verbose" in mission else False
    def set_model_name(self, model_name: str):
        """Set the model name before model creation"""
        self.model_name = model_name
    def analyze_image_full(
        self,
        image: Union[str, Image.Image, list],
        prompt: str,
        system_prompt_type: str = None,
    ):
        """
        Analyze one or more images with the given prompt, returning full response object.

        Args:
            image: Single image (PIL Image, file path, or URL) or list of images
            prompt: Text prompt for analysis

        Returns:
            Analysis result as full response object (content + metadata)
        """
        model = self.create_model()

        # Create message content starting with prompt
        content = [{"type": "text", "text": prompt}]
        
        # Add image(s) to content
        if isinstance(image, list):
            for img in image:
                content.append(self._image_to_content(img))
        else:
            content.append(self._image_to_content(image))


        # Create message list and invoke model
        messages = [HumanMessage(content=content)]
        if isinstance(system_prompt_type, str):
            system_prompt = generate_prompt(prompt_type=system_prompt_type, mission=self.mission["mission"])
            messages.insert(0, SystemMessage(content=system_prompt.strip()))

        return model.invoke(messages)

    def _image_to_content(self, image: Union[str, Image.Image]) -> dict:
        """Convert image to content format for vision models"""
        is_bedrock_anthropic = self.model_name and self.model_name.startswith("us.anthropic")
       
        if isinstance(image, pathlib.PosixPath):
            image = str(image)
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                return {"type": "image_url", "image_url": {"url": image}}
            else:
                with open(image, "rb") as f:
                    image_data = f.read()
                mime_type = "image/png" if image.lower().endswith(".png") else "image/jpeg"
                b64_string = base64.b64encode(image_data).decode()
 
                if is_bedrock_anthropic and len(b64_string) > MAX_B64_BYTES:
                    print(f"⚠ Image exceeds Bedrock limit ({len(b64_string)} bytes b64), compressing to JPEG...")
                    pil_img = Image.open(image).convert("RGB")
                    b64_string, mime_type = self._compress_image(pil_img)
                    print(f"  ✓ Compressed to {len(b64_string)} bytes b64")
 
                return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_string}"}}
 
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=90)
            b64_string = base64.b64encode(buffer.getvalue()).decode()
 
            if is_bedrock_anthropic and len(b64_string) > MAX_B64_BYTES:
                print(f"⚠ PIL image exceeds Bedrock limit ({len(b64_string)} bytes b64), compressing to JPEG...")
                b64_string, _ = self._compress_image(image)
                print(f"  ✓ Compressed to {len(b64_string)} bytes b64")
 
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_string}"}}
 
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
    def _compress_image(self, image: Image.Image) -> tuple:
        """Progressively compress image to fit under Bedrock's 5MB base64 limit."""
        image = image.convert("RGB")
        for quality in [85, 75, 60, 45]:
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=quality)
            b64 = base64.b64encode(buf.getvalue()).decode()
            if len(b64) < MAX_B64_BYTES:
                return b64, "image/jpeg"
        # Last resort: resize then compress
        image.thumbnail((1568, 1568), Image.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"
       
    def _save_prompt_to_disk(self, base_dir: str, prompt_type: str, step_index: int, prompt_text: str) -> None:
        """Save prompt text to a file without storing in memory."""
        try:
            target_dir = Path(base_dir) / "debug_outputs"
            target_dir.mkdir(parents=True, exist_ok=True)
            file_path = target_dir / f"step_{step_index}_{prompt_type}_prompt.txt"
            
            with open(file_path, "w") as f:
                f.write(prompt_text)
        except Exception as exc:
            print(f"⚠ Failed to save prompt: {exc}")    
            
    def save_prompt_to_disk(self, file_name, prompt_text: str, char_limit: int = 80) -> None:
        """save the prompt by wrapping the text to a fixed number of characters per line"""
        try:
            wrapped_prompt = "\n".join([prompt_text[i:i+char_limit] for i in range(0, len(prompt_text), char_limit)])
            file_path = Path(file_name)
            # check if parent directory exists, if not create it
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "w") as f:
                f.write(wrapped_prompt)
        except Exception as exc:
            print(f"⚠ Failed to save full prompt: {exc}")
        
    def create_model(self):
        """
        Automatically select model provider based on model_name.

        Supported providers:
        - OpenAI: "gpt-4o", "gpt-4o-mini", etc.
        - Anthropic: "claude-3-5-sonnet-20241022", etc.
        - Google Gemini/Gemma: "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp", "gemma-4-31b", etc.
        - AWS Bedrock: "amazon.nova-pro-v1:0", "anthropic.claude-3-5-sonnet-20241022-v2:0", etc.
          (For Anthropic models on Bedrock, use model_provider="bedrock")
        - vLLM: "vllm-<model_name>" (custom local inference server)

        Examples:
            >>> model = create_model("gpt-4o")  # OpenAI
            >>> model = create_model("claude-3-5-sonnet-20241022")  # Anthropic
            >>> model = create_model("gemini-1.5-pro")  # Google Gemini
            >>> model = create_model("gemma-4-31b")  # Google Gemma
            >>> model = create_model("amazon.nova-pro-v1:0")  # Bedrock (auto-inferred)
            >>> model = create_model("vllm-Llama-3-8B")  # Local vLLM server
        """
        # All major providers: OpenAI, Anthropic, Bedrock, Google, etc.
        # API keys are automatically read from environment variables
        if not self.model_name:
            raise ValueError("model_name must be set before creating model")
        # try:
            
        normalized_model_name = self.model_name.strip()
        normalized_model_name_lower = normalized_model_name.lower()

        models_related = self.mission.get("mission", {}).get("models_related", {}) if self.mission else {}

        temperature = models_related.get("temperature")
        assert isinstance(temperature, (int, float)), "Temperature must be a number"
        top_p = float(models_related.get("top_p", 0.95))
        top_k = int(models_related.get("top_k", 20))
        min_p = float(models_related.get("min_p", 0.0))
        presence_penalty = float(models_related.get("presence_penalty", 1.5))
        repetition_penalty = float(models_related.get("repetition_penalty", 1.0))
        if self.verbose:
            print(
                f"   🤖 Initializing model '{normalized_model_name}' with "
                f"temperature={temperature}, top_p={top_p}, top_k={top_k}, min_p={min_p}, "
                f"presence_penalty={presence_penalty}, repetition_penalty={repetition_penalty}..."
            )
        # For Anthropic models on Bedrock, explicitly specify provider
        aws_prefixes = ("amazon.", "anthropic.", "us")
        if any(normalized_model_name_lower.startswith(i) for i in aws_prefixes):
            # This is a Bedrock model (format: "anthropic.claude-3-...")
            return init_chat_model(normalized_model_name, model_provider="bedrock", temperature=0)
        elif normalized_model_name_lower.startswith("gpt-") or normalized_model_name_lower.startswith("openai-"):
            return ChatOpenAI(model=normalized_model_name, temperature=temperature if self.mission else 0)
        elif normalized_model_name_lower.startswith("claude-") or normalized_model_name_lower.startswith("anthropic-"):
            return ChatAnthropic(model=normalized_model_name, temperature=temperature if self.mission else 0)
        elif (
            normalized_model_name_lower.startswith("gemini-")
            or normalized_model_name_lower.startswith("google-")
            or normalized_model_name_lower.startswith("gemma-")
        ):
            # Use Google GenAI API (requires GOOGLE_API_KEY or GEMINI_API_KEY).
            # Valid models include Gemini and Gemma families.
            # Normalize to lowercase model IDs accepted by Google endpoints.
            google_model_name = normalized_model_name_lower
            if google_model_name.startswith("models/"):
                google_model_name = google_model_name[len("models/"):]
            thinking_level = models_related.get("thinking_level")
            if isinstance(thinking_level, str):
                normalized_thinking_level = thinking_level.strip().lower()
                if normalized_thinking_level in {"", "none", "null"}:
                    thinking_level = None
                else:
                    thinking_level = normalized_thinking_level
            if thinking_level is not None:
                temperature=temperature if self.mission else 0
                print(f"   🧠 Setting thinking_level='{thinking_level}' with temperature={temperature} for Google model '{google_model_name}'.")
                return ChatGoogleGenerativeAI(
                    model=google_model_name,
                    temperature=temperature,
                    model_kwargs={
                        "thinking_config": {
                            "thinking_level": thinking_level  # "low" | "medium" | "high"
                        }
                    },
                )
            return ChatGoogleGenerativeAI(model=google_model_name, temperature=temperature if self.mission else 0)
        elif normalized_model_name_lower.startswith("vertex-"):
            # Use Vertex AI (requires GCP authentication via gcloud)
            # Model name format: "vertex-gemini-pro" -> "gemini-pro"
            vertex_model_name = normalized_model_name[len("vertex-"):]
            return ChatVertexAI(model=vertex_model_name, temperature=temperature if self.mission else 0)

        # Special case: vLLM (not supported by init_chat_model)
        elif normalized_model_name_lower.startswith("vllm-"):
            inference_server_url = "http://localhost:8000/v1"
            local_model_name = normalized_model_name[len("vllm-"):]
            return ChatOpenAI(
                model=local_model_name,
                openai_api_key="EMPTY",
                openai_api_base=inference_server_url,
                temperature=0,
            )
        elif normalized_model_name_lower.startswith("llama-cpp-"):
            inference_server_url = "http://127.0.0.1:8088"
            local_model_name = normalized_model_name[len("llama-cpp-"):]
            
            # Use custom wrapper to handle reasoning content correctly
            return LocalLlamaWrapper(
                model_name=local_model_name,
                base_url=inference_server_url,
                temperature=temperature if self.mission else 0,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                max_tokens=4096,
                stop=["<|im_end|>", "<|endoftext|>", "END_OF_TURN"],
                verbose=self.verbose
            )
        elif normalized_model_name_lower.startswith("qwen3."):
            return ChatOpenAI(
                model=normalized_model_name,
                openai_api_key=os.getenv("DASHSCOPE_API_KEY"), 
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                temperature=temperature if self.mission else 0,
            )
        else:
            # Auto-infer provider for all other models
            return init_chat_model(normalized_model_name, temperature=0)
        
        # except Exception as e:
        #     raise ValueError(
        #         f"Failed to initialize model '{self.model_name}'. "
        #         f"Make sure the model name is valid and required API keys/credentials are set in .env file. "
        #         f"Supported providers: OpenAI (gpt-*), Anthropic (claude-*), Google Gemini (gemini-*), "
        #         f"AWS Bedrock (amazon.*, anthropic.*), vLLM (vllm-*). "
        #         f"Error: {str(e)}"
        #     )

    def validate_response(self, response):
        """
        Cleaner for VLM responses.
        Handles cases where models return Lists instead of Strings.
        """
        # LangChain providers may return structured message parts instead of plain strings.
        # Normalize those into clean text to keep downstream parsers stable.
        if isinstance(response, dict):
            if "text" in response and isinstance(response["text"], str):
                response = response["text"]
            else:
                response = str(response)

        elif isinstance(response, list):
            text_chunks = []
            for item in response:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_chunks.append(item["text"])
                elif isinstance(item, str):
                    text_chunks.append(item)
                else:
                    text_chunks.append(str(item))
            response = "\n".join([chunk for chunk in text_chunks if chunk and chunk.strip()])

        elif not isinstance(response, str):
            response = str(response)

        return response.strip()

class LocalLlamaWrapper:
    def __init__(
        self,
        model_name,
        base_url="http://127.0.0.1:8088/v1",
        temperature=0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_tokens=4096,
        stop=None,
        verbose=False,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.stop = stop or []
        self.verbose = verbose
        try:
             import requests
        except ImportError:
             raise ImportError("requests library is required for LocalLlamaWrapper")

    def invoke(self, messages: list) -> HumanMessage: # Returns Message object compatible with LangChain
        import requests
        
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            role = "user"
            content = msg.content
            if hasattr(msg, "role"):
                if msg.role == "system": role = "system"
                elif msg.role == "assistant": role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            
            # Handle list content (multimodal)
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, dict):
                         if "image_url" in item or "text" in item or item.get("type") in ["text", "image_url"]:
                             new_content.append(item)
                    elif isinstance(item, str):
                        new_content.append({"type": "text", "text": item})
                content = new_content

            openai_messages.append({"role": role, "content": content})

        payload = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
            "stream": False,
            "stop": self.stop
        }
        
        try:
            import requests

            # --- Reasoning Budget Implementation ---
            MAX_REASONING_CHARS = 4000  # Approx 1000 tokens
       
            payload["stream"] = True

            if self.verbose:
                 print(f"[LocalLlamaWrapper] Sending streaming request to {self.base_url}/chat/completions")
                 print(
                     "[LocalLlamaWrapper] Sampling params: "
                     f"temperature={self.temperature}, top_p={self.top_p}, top_k={self.top_k}, "
                     f"min_p={self.min_p}, presence_penalty={self.presence_penalty}, "
                     f"repetition_penalty={self.repetition_penalty}"
                 )
            
            # Start streaming request
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=300,
                    stream=True
                )
                if response.status_code != 200:
                    print(f"[LocalLlamaWrapper] Error Status: {response.status_code}")
                    print(f"[LocalLlamaWrapper] Error Response: {response.text}")
                response.raise_for_status()
            except requests.exceptions.RequestException as req_err:
                 print(f"[LocalLlamaWrapper] Request Failed: {req_err}")
                 if hasattr(req_err, 'response') and req_err.response is not None:
                      print(f"[LocalLlamaWrapper] Server Response: {req_err.response.text}")
                 raise req_err

            # Tracking variables
            full_content = ""
            full_reasoning = ""
            finish_reason = None
            token_usage = None
            thinking_timeout_triggered = False

            import json
            for line in response.iter_lines():
                if not line: continue
                line_text = line.decode("utf-8")
                if not line_text.startswith("data: "): continue
                
                json_str = line_text[6:]
                if json_str.strip() == "[DONE]": break
                
                try:
                    chunk = json.loads(json_str)
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})
                    
                    # Accumulate content
                    content_chunk = delta.get("content", "")
                    reasoning_chunk = delta.get("reasoning_content", "")
                    
                    if content_chunk: full_content += content_chunk
                    if reasoning_chunk: full_reasoning += reasoning_chunk
                    
                    finish_reason = choice.get("finish_reason")
                    
                    # Usage stats usually come at the end
                    if "usage" in chunk and chunk["usage"]:
                         token_usage = chunk["usage"]

                    # Check reasoning budget
                    if len(full_reasoning) > MAX_REASONING_CHARS and not thinking_timeout_triggered:
                        print(f"[LocalLlamaWrapper] ⚠️ Reasoning budget exceeded ({len(full_reasoning)} chars). Forcing stop.")
                        thinking_timeout_triggered = True
                        
                        # Instead of closing the response immediately, we break the loop. 
                        # The connection will be closed when the loop exits, or when requests cleans up.
                        # However, requests.iter_lines() cleans up if you stop iterating.
                        # For robustness, we will not call response.close() explicitly here, 
                        # relying on the context manager/GC or let the iteration finish gracefully if we could, but we can't.
                        # Breaking the loop stops reading the stream.
                        
                        # Wait a bit before potentially killing the connection? 
                        # No, requests handles closing the socket.
                        break
                        
                except json.JSONDecodeError:
                    continue
            
            # Ensure the connection is cleaned up if we broke out of the loop
            response.close()
            
            # Add a small delay to allow the server to recover from the interrupted stream
            import time
            time.sleep(1.0) # 1 second cool-down

            # If stopped due to thinking budget
            if thinking_timeout_triggered:
                print(f"[LocalLlamaWrapper] Budget exceeded. Returning captured reasoning ({len(full_reasoning)} chars)...")
                # Fallback: Just return what we have to avoid crashing server on retry
                full_content = f"{full_reasoning}\n\n[Reasoning truncated due to length limit. No final answer provided.]"
                
                # Note: The downstream parser will likely fail if it expects strict XML, 
                # but this is better than a 500 error crashing the whole script.
                # Ideally, we should construct a fake XML answer here.
                if "</think>" not in full_content:
                     full_content += "\n</think>"
                # Append a fallback answer
                full_content += "\n<answer>FAILED_DUE_TO_LENGTH</answer>"

            # Helper to create AIMessage-like object
            class LlamaResponse:
                def __init__(self, content, additional_kwargs, response_metadata):
                    self.content = content
                    self.additional_kwargs = additional_kwargs
                    self.response_metadata = response_metadata
                def __repr__(self):
                    c_peek = self.content[:50] if self.content else ""
                    r_peek = self.additional_kwargs.get('reasoning_content', '')[:50] if self.additional_kwargs else ""
                    return f"LlamaResponse(content={c_peek}..., reasoning={r_peek}...)"

            # If content is empty but reasoning exists, use reasoning as primary content but keep copy
            if not full_content and full_reasoning:
                full_content = full_reasoning
                if self.verbose:
                     print(f"[LocalLlamaWrapper] Promoting reasoning to content ({len(full_reasoning)} chars)")

            return LlamaResponse(
                content=full_content,
                additional_kwargs={"reasoning_content": full_reasoning},
                response_metadata={"finish_reason": finish_reason, "token_usage": token_usage}
            )
            
        except Exception as e:
            print(f"[LocalLlamaWrapper] Error: {e}")
            raise e