import os
from typing import Any, Dict, List
from openai import OpenAI
from .base_provider import BaseLLMProvider
import json


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self.model = None

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    def validate_config(self) -> bool:
        """Validate OpenAI configuration"""
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.strip() == "":
            self.log_error("OPENAI_API_KEY not found in environment or config")
            return False
        return True

    def initialize(self) -> None:
        """Initialize OpenAI client"""
        if not self.validate_config():
            raise ValueError("Invalid OpenAI configuration")

        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = self.config.get("base_url") or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.model = self.config.get("model") or os.getenv("OPENAI_MODEL", "gpt-4-0613")

        self.log_info(f"Initializing with base URL: {base_url}")
        self.log_info(f"Using model: {self.model}")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.log_info("Client initialized successfully")

    def generate_response(self, messages: List[Dict], functions: List[Dict]) -> Dict:
        """Generate response using OpenAI API"""
        if not self.client:
            raise RuntimeError("Provider not initialized")

        managed_messages = self.manage_context_size(messages, functions)

        # Cap output tokens to control costs
        max_output_tokens = None
        try:
            max_output_tokens = (
                int(
                    self.config.get("max_output_tokens")
                    or os.getenv("OPENAI_MAX_OUTPUT_TOKENS")
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

        self.log_debug(
            f"Sending {len(managed_messages)} messages to OpenAI (managed from {len(messages)})"
        )

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
            self.log_info("Received response from OpenAI")

            # Log actual usage if available
            try:
                usage = getattr(response, "usage", None)
                if usage:
                    self.log_info(
                        f"OpenAI usage - prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens}, total: {usage.total_tokens}"
                    )
            except Exception:
                pass

            if message.get("function_call"):
                self.log_info(
                    f"Function call requested: {message['function_call']['name']}"
                )

            return message

        except Exception as e:
            error_msg = str(e)
            self.log_error(f"OpenAI API error: {error_msg}")

            if "context length" in error_msg.lower() or "token" in error_msg.lower():
                self.log_warning(
                    "Token limit error detected, trying aggressive compression..."
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
                        "Received response from OpenAI after emergency compression"
                    )
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
