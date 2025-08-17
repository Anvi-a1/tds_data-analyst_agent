import os
import json
from typing import Any, Dict, List, Optional, Tuple

from .base_provider import BaseLLMProvider
import json


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini provider implementation.

    This provider handles the conversion of a generic message format (similar to
    OpenAI's) to the format required by the google-genai SDK. It supports
    system instructions, text generation, and tool/function calling.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name: Optional[str] = None
        self.client: Optional[Any] = None

    @property
    def provider_name(self) -> str:
        return "Gemini"

    def validate_config(self) -> bool:
        """Validate that the necessary Gemini API key is available."""
        api_key = self.config.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.log_error(
                "GEMINI_API_KEY not found in environment variables or config."
            )
            return False
        return True

    def initialize(self) -> None:
        """
        Initializes the Gemini client.

        Raises:
            ValueError: If the configuration is invalid.
            ImportError: If the 'google-genai' package is not installed.
        """
        if not self.validate_config():
            raise ValueError("Invalid Gemini configuration")

        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "The 'google-genai' package is not installed. "
                "Please install it using `pip install google-genai`."
            )

        api_key = self.config.get("api_key") or os.getenv("GEMINI_API_KEY")
        self.model_name = self.config.get("model") or os.getenv(
            "GEMINI_MODEL", "gemini-1.5-flash-latest"
        )

        self.client = genai.Client(api_key=api_key)
        self.log_info(f"Gemini provider initialized with model: {self.model_name}")

    def _prepare_request_data(
        self, messages: List[Dict]
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Converts a standard message list to Gemini's `contents` format and extracts the system instruction.

        Args:
            messages: A list of message dictionaries in a standard format.

        Returns:
            A tuple containing the list of contents for the API call and the system instruction string.
        """
        gemini_contents = []
        system_instruction = None

        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                system_instruction = msg["content"]
                break

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                continue

            gemini_role = "model" if role == "assistant" else role

            if role == "tool":
                gemini_contents.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "function_response": {
                                    "name": msg.get("name"),
                                    "response": {"content": msg.get("content")},
                                }
                            }
                        ],
                    }
                )
            elif msg.get("function_call"):
                # Include assistant function call turn so the next tool response directly follows it
                fc = msg["function_call"]
                try:
                    args = fc.get("arguments")
                    args_obj = (
                        json.loads(args) if isinstance(args, str) else (args or {})
                    )
                except Exception:
                    args_obj = {}
                gemini_contents.append(
                    {
                        "role": gemini_role,
                        "parts": [
                            {
                                "function_call": {
                                    "name": fc.get("name"),
                                    "args": args_obj,
                                }
                            }
                        ],
                    }
                )
            elif msg.get("tool_calls"):
                # OpenAI-style multiple tool calls. Encode each as a function_call part in a single assistant turn.
                parts = []
                for tc in msg.get("tool_calls", []) or []:
                    fn = (tc or {}).get("function") or {}
                    try:
                        args = fn.get("arguments")
                        args_obj = (
                            json.loads(args) if isinstance(args, str) else (args or {})
                        )
                    except Exception:
                        args_obj = {}
                    parts.append(
                        {
                            "function_call": {
                                "name": fn.get("name"),
                                "args": args_obj,
                            }
                        }
                    )
                if parts:
                    gemini_contents.append({"role": gemini_role, "parts": parts})
            elif isinstance(msg.get("content"), str):
                gemini_contents.append(
                    {"role": gemini_role, "parts": [{"text": msg["content"]}]}
                )

        return gemini_contents, system_instruction

    def generate_response(
        self, messages: List[Dict], functions: List[Dict] = None
    ) -> Dict:
        """
        Generate a response using the Gemini API.

        This method supports both text and function-calling responses and is designed
        to handle a more robust, OpenAI-like message format.

        Args:
            messages: A list of message dictionaries representing the conversation history.
            functions: A list of function definitions available to the model.

        Returns:
            A dictionary representing the model's response message.
        """
        if not self.client or not self.model_name:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        # Manage context size to prevent token limit errors
        managed_messages = self.manage_context_size(messages, functions)

        gemini_tools = functions or []
        gemini_contents, system_instruction = self._prepare_request_data(
            managed_messages
        )

        self.log_debug(
            f"Sending {len(gemini_contents)} content blocks to Gemini (managed from {len(messages)} messages)"
        )

        # Cap output tokens if environment provides a hint (Gemini uses safety configs, not exact cap here)
        max_output_tokens = None
        try:
            max_output_tokens = (
                int(
                    self.config.get("max_output_tokens")
                    or os.getenv("GEMINI_MAX_OUTPUT_TOKENS")
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

        try:
            # Use the updated google-genai client interface with function calling.
            from google.genai import types

            tool_obj = types.Tool(function_declarations=gemini_tools)
            config = types.GenerateContentConfig(
                tools=[tool_obj],
                system_instruction=system_instruction,
                # token limits are hints; set only if available
                max_output_tokens=max_output_tokens if max_output_tokens else None,
            )

            # Directly call generate_content on client.models with the config.
            response = self.client.models.generate_content(
                model=self.model_name, contents=gemini_contents, config=config
            )

        except Exception as e:
            error_msg = str(e)
            self.log_error(
                f"An unexpected error occurred while calling the Gemini API: {error_msg}"
            )

            if (
                "token" in error_msg.lower() and "exceeds" in error_msg.lower()
            ) or "INVALID_ARGUMENT" in error_msg:
                self.log_warning(
                    "Token limit error detected, trying emergency compression..."
                )
                try:
                    emergency_messages = []
                    if managed_messages and managed_messages[0].get("role") == "system":
                        system_content = (
                            managed_messages[0].get("content", "")[:1500]
                            + "\n[System message truncated for space]"
                        )
                        emergency_messages.append(
                            {"role": "system", "content": system_content}
                        )

                    # Keep only the last user message
                    for msg in reversed(managed_messages):
                        if msg.get("role") == "user":
                            user_content = msg.get("content", "")
                            if len(user_content) > 8000:
                                user_content = (
                                    user_content[:8000]
                                    + "\n[Message truncated due to length constraints]"
                                )
                            emergency_messages.append(
                                {"role": "user", "content": user_content}
                            )
                            break

                    self.log_info(
                        f"Emergency compression: using {len(emergency_messages)} messages"
                    )

                    gemini_contents, system_instruction = self._prepare_request_data(
                        emergency_messages
                    )

                    response = self.client.models.generate_content(
                        model=self.model_name, contents=gemini_contents, config=config
                    )

                    self.log_info(
                        "Received response from Gemini after emergency compression"
                    )

                except Exception as e2:
                    self.log_error(f"Emergency compression also failed: {str(e2)}")
                    return {
                        "role": "assistant",
                        "content": f"I apologize, but the context is too large for me to process. Please try with a smaller request or less data. Error: {error_msg}",
                    }
            else:
                return {
                    "role": "assistant",
                    "content": f"Sorry, an error occurred with the AI provider: {error_msg}",
                }

        self.log_info("Received response from Gemini.")

        final_message = {"role": "assistant", "content": None}

        # If there are no candidates at all, it's likely blocked or empty
        if not getattr(response, "candidates", None):
            feedback = getattr(response, "prompt_feedback", None)
            reason = (
                feedback.block_reason.name
                if feedback and getattr(feedback, "block_reason", None)
                else "Unknown"
            )
            self.log_error(
                f"Request was blocked or returned no candidates. Reason: {reason}"
            )
            final_message["content"] = (
                f"Your request was blocked for safety reasons ({reason}). Please adjust your prompt."
            )
            return final_message

        # Robustly extract either a function call or text from candidates/parts
        func_call = None
        text_out = None

        # Some SDK responses expose an aggregated .text
        try:
            aggregated_text = getattr(response, "text", None)
            if aggregated_text:
                text_out = aggregated_text
        except Exception:
            pass

        if not (func_call or text_out):
            try:
                for cand in response.candidates:
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    parts = getattr(content, "parts", None) or []
                    if not parts:
                        continue
                    for part in parts:
                        # Prefer function call if present
                        if hasattr(part, "function_call") and getattr(
                            part, "function_call"
                        ):
                            func_call = part.function_call
                            break
                        # Otherwise take first text
                        if hasattr(part, "text") and getattr(part, "text"):
                            text_out = part.text
                            # Don't break here yet; a later part may have a function call
                    if func_call or text_out:
                        break
            except Exception as e:
                self.log_warning(
                    f"Failed to parse candidates/parts structure safely: {e}"
                )

        # Build final message based on what we found
        if func_call:
            try:
                final_message["tool_calls"] = [
                    {
                        "id": f"call_{os.urandom(8).hex()}",
                        "type": "function",
                        "function": {
                            "name": func_call.name,
                            "arguments": json.dumps(dict(func_call.args)),
                        },
                    }
                ]
                self.log_info(f"Function call requested: {func_call.name}")
            except Exception as e:
                # If something goes wrong forming the function call, fall back to text
                self.log_error(f"Error extracting function call from response: {e}")
                if text_out:
                    final_message["content"] = text_out
                else:
                    final_message["content"] = ""
        else:
            if text_out is not None:
                final_message["content"] = text_out
                self.log_info("Text response received from Gemini.")
            else:
                # Nothing parseable; return an empty string with a warning to avoid crashing
                self.log_warning(
                    "Gemini response had candidates but no content parts or text; returning empty content."
                )
                final_message["content"] = ""

        # print(response)
        return final_message
