import os
import json
from openai import OpenAI
from typing import Any, Dict, List
from .base_provider import BaseLLMProvider


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self.model = None

    @property
    def provider_name(self) -> str:
        return "OpenRouter"

    def validate_config(self) -> bool:
        """Validate OpenRouter configuration"""
        api_key = self.config.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            self.log_error("OPENROUTER_API_KEY not found in environment or config")
            return False
        return True

    def initialize(self) -> None:
        """Initialize OpenRouter client"""
        if not self.validate_config():
            raise ValueError("Invalid OpenRouter configuration")

        api_key = self.config.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        base_url = self.config.get("base_url") or os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.model = self.config.get("model") or os.getenv(
            "OPENROUTER_MODEL", "openai/gpt-4-turbo"
        )

        headers = {
            "HTTP-Referer": self.config.get("app_name", "Data-Analyst-Agent"),
            "X-Title": self.config.get("app_title", "Data Analyst Agent"),
        }

        self.log_info(f"Initializing with base URL: {base_url}")
        self.log_info(f"Using model: {self.model}")

        self.client = OpenAI(
            api_key=api_key, base_url=base_url, default_headers=headers
        )
        self.log_info("Client initialized successfully")

    def generate_response(self, messages: List[Dict], functions: List[Dict]) -> Dict:
        """Generate response using OpenRouter API"""
        if not self.client:
            raise RuntimeError("Provider not initialized")

        managed_messages = self.manage_context_size(messages, functions)

        self.log_debug(
            f"Sending {len(managed_messages)} messages to OpenRouter (managed from {len(messages)})"
        )

        # Cap output tokens to control costs
        max_output_tokens = None
        try:
            max_output_tokens = (
                int(
                    self.config.get("max_output_tokens")
                    or os.getenv("OPENROUTER_MAX_OUTPUT_TOKENS")
                    or os.getenv("LLM_MAX_OUTPUT_TOKENS", "0")
                )
                or None
            )
        except Exception:
            max_output_tokens = None

        # Log estimated prompt tokens
        try:
            self._ensure_token_manager()
            prompt_tokens_est = self.token_manager.estimate_messages_tokens(
                managed_messages
            )
            func_tokens_est = (
                self.token_manager.estimate_tokens(json.dumps(functions))
                if functions
                else 0
            )
            self.log_info(
                f"Estimated prompt tokens: {prompt_tokens_est} (+{func_tokens_est} for tools); max_output_tokens={max_output_tokens}"
            )
        except Exception:
            pass

        # Use modern tool-calling always; do not use legacy functions/function_call
        tools = [{"type": "function", "function": f} for f in (functions or [])] or None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=managed_messages,
                tools=tools,
                tool_choice="auto" if tools else None,
                temperature=0,
                max_tokens=max_output_tokens,
            )

            message = response.choices[0].message.model_dump()
            # print(response)
            self.log_info("Received response from OpenRouter")

            if message.get("tool_calls"):
                self.log_info(
                    f"Tool call requested: {message['tool_calls'][0]['function']['name']}"
                )
            elif message.get("function_call"):
                self.log_info(
                    f"Function call requested: {message['function_call']['name']}"
                )

            return message

        except Exception as e:
            error_msg = str(e)
            self.log_error(f"OpenRouter API error: {error_msg}")

            if "context length" in error_msg.lower() or "token" in error_msg.lower():
                self.log_warning(
                    "Token limit error detected, trying emergency compression..."
                )
                try:
                    emergency_messages = []
                    if managed_messages and managed_messages[0].get("role") == "system":
                        system_content = (
                            managed_messages[0].get("content", "")[:2000]
                            + "\n[System message truncated]"
                        )
                        emergency_messages.append(
                            {"role": "system", "content": system_content}
                        )

                    for msg in reversed(managed_messages):
                        if msg.get("role") == "user":
                            user_content = msg.get("content", "")
                            if len(user_content) > 10000:
                                user_content = (
                                    user_content[:10000]
                                    + "\n[Message truncated due to length]"
                                )
                            emergency_messages.append(
                                {"role": "user", "content": user_content}
                            )
                            break

                    self.log_info(
                        f"Emergency compression: using {len(emergency_messages)} messages"
                    )

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=emergency_messages,
                        tools=tools,
                        tool_choice="auto" if tools else None,
                        temperature=0,
                        max_tokens=max_output_tokens,
                    )

                    message = response.choices[0].message.model_dump()
                    self.log_info(
                        "Received response from OpenRouter after emergency compression"
                    )
                    # print(response)
                    return message

                except Exception as e2:
                    self.log_error(f"Emergency compression also failed: {str(e2)}")
                    return {
                        "role": "assistant",
                        "content": f"I apologize, but the context is too large for me to process. Please try with a smaller request or less data. Error: {error_msg}",
                    }

            return {
                "role": "assistant",
                "content": f"I encountered an error: {error_msg}. Please try again.",
            }
