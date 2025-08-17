import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class TokenManager:
    """Manages token limits and context truncation for LLM providers"""

    TOKEN_LIMITS = {
        "gpt-4": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-4.1-nano": 1047576,
        "gpt-3.5-turbo": 16385,
        "gemini-1.5-flash": 1048576,
        "gemini-1.5-pro": 2097152,
        "gemini-2.5-flash": 1048576,
    }

    RESPONSE_RESERVE = 4000
    FUNCTION_RESERVE = 2000
    SAFETY_MARGIN = 1000

    def __init__(self, model_name: str = None):
        self.model_name = model_name or "gpt-4"
        self.max_tokens = self.get_token_limit(self.model_name)
        self.available_tokens = (
            self.max_tokens
            - self.RESPONSE_RESERVE
            - self.FUNCTION_RESERVE
            - self.SAFETY_MARGIN
        )
        logger.info(
            f"TokenManager initialized for {self.model_name}: {self.max_tokens} max, {self.available_tokens} available"
        )

    def get_token_limit(self, model_name: str) -> int:
        """Get token limit for a specific model"""
        if model_name in self.TOKEN_LIMITS:
            return self.TOKEN_LIMITS[model_name]

        for known_model, limit in self.TOKEN_LIMITS.items():
            if known_model in model_name.lower():
                return limit

        logger.warning(f"Unknown model {model_name}, using conservative limit of 32000")
        return 32000

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using a simple approximation.
        This is faster than tiktoken but less accurate.
        Roughly: 1 token ≈ 4 characters for English text
        """
        if not text:
            return 0
        return len(text) // 3

    def estimate_message_tokens(self, message: Dict) -> int:
        """Estimate tokens for a single message"""
        tokens = 0

        tokens += 4

        content = message.get("content", "")
        if isinstance(content, str):
            tokens += self.estimate_tokens(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    tokens += self.estimate_tokens(json.dumps(item))
                else:
                    tokens += self.estimate_tokens(str(item))

        if message.get("function_call"):
            tokens += self.estimate_tokens(json.dumps(message["function_call"]))

        if message.get("tool_calls"):
            tokens += self.estimate_tokens(json.dumps(message["tool_calls"]))

        return tokens

    def estimate_messages_tokens(self, messages: List[Dict]) -> int:
        """Estimate total tokens for a list of messages"""
        return sum(self.estimate_message_tokens(msg) for msg in messages)

    def truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within max_tokens"""
        estimated_tokens = self.estimate_tokens(content)

        if estimated_tokens <= max_tokens:
            return content

        target_chars = max_tokens * 3

        if len(content) <= target_chars:
            return content

        truncated = content[:target_chars]

        for break_point in ["\n\n", "\n", ". ", ", ", " "]:
            last_break = truncated.rfind(break_point)
            if last_break > target_chars * 0.8:
                truncated = truncated[: last_break + len(break_point)]
                break

        truncated += "\n\n[... content truncated due to length ...]"
        logger.info(f"Truncated content from {len(content)} to {len(truncated)} chars")
        return truncated

    def compress_large_data(self, content: str, max_tokens: int) -> str:
        """Compress large structured data by removing redundant information"""
        if self.estimate_tokens(content) <= max_tokens:
            return content

        if "<table" in content.lower() and "</table>" in content.lower():
            return self._compress_html_table(content, max_tokens)
        elif content.strip().startswith("[") or content.strip().startswith("{"):
            return self._compress_json_data(content, max_tokens)
        else:
            return self.truncate_content(content, max_tokens)

    def _compress_html_table(self, content: str, max_tokens: int) -> str:
        """Compress HTML table by keeping structure but reducing rows"""
        lines = content.split("\n")

        table_start = next(
            (i for i, line in enumerate(lines) if "<table" in line.lower()), None
        )
        table_end = next(
            (i for i, line in enumerate(lines) if "</table>" in line.lower()), None
        )

        if table_start is None or table_end is None:
            return self.truncate_content(content, max_tokens)

        header_end = None
        for i in range(table_start, min(table_start + 20, len(lines))):
            if "</thead>" in lines[i].lower() or "<tbody>" in lines[i].lower():
                header_end = i
                break

        if header_end:
            header_part = lines[: header_end + 1]

            data_start = header_end + 1
            data_end = table_end

            if data_end - data_start > 20:
                first_rows = lines[data_start : data_start + 10]
                last_rows = lines[data_end - 5 : data_end]
                sample_note = [
                    "<tr><td colspan='100%'>[... {} rows omitted ...]</td></tr>".format(
                        data_end - data_start - 15
                    )
                ]

                compressed = (
                    header_part
                    + first_rows
                    + sample_note
                    + last_rows
                    + lines[table_end:]
                )
                result = "\n".join(compressed)

                if self.estimate_tokens(result) <= max_tokens:
                    return result

        return self.truncate_content(content, max_tokens)

    def _compress_json_data(self, content: str, max_tokens: int) -> str:
        """Compress JSON data by sampling"""
        try:
            data = json.loads(content)

            if isinstance(data, list) and len(data) > 10:
                compressed = (
                    data[:5]
                    + [{"...": f"{len(data) - 10} items omitted..."}]
                    + data[-5:]
                )
                result = json.dumps(compressed, indent=2)

                if self.estimate_tokens(result) <= max_tokens:
                    return result

        except json.JSONDecodeError:
            pass

        return self.truncate_content(content, max_tokens)

    def manage_conversation_context(
        self, messages: List[Dict], functions: List[Dict] = None
    ) -> List[Dict]:
        """
        Manage conversation context to stay within token limits.
        Returns truncated/compressed messages that fit within limits.
        """
        function_tokens = 0
        if functions:
            function_tokens = self.estimate_tokens(json.dumps(functions))

        available_for_messages = self.available_tokens - function_tokens

        if available_for_messages < 1000:
            logger.error("Not enough tokens available for meaningful conversation")
            return messages[-2:]

        current_tokens = self.estimate_messages_tokens(messages)

        if current_tokens <= available_for_messages:
            logger.debug(
                f"Context fits: {current_tokens}/{available_for_messages} tokens"
            )
            return messages

        logger.warning(
            f"Context too large: {current_tokens}/{available_for_messages} tokens. Compressing..."
        )

        if not messages:
            return messages

        system_msg = None
        if messages[0].get("role") == "system":
            system_msg = messages[0]
            messages = messages[1:]

        if not messages:
            return [system_msg] if system_msg else []

        last_user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg
                break

        compressed_messages = []

        if system_msg:
            sys_tokens = self.estimate_message_tokens(system_msg)
            if sys_tokens > available_for_messages // 4:
                system_content = system_msg.get("content", "")
                compressed_content = self.truncate_content(
                    system_content, available_for_messages // 4
                )
                system_msg = {**system_msg, "content": compressed_content}
            compressed_messages.append(system_msg)

        used_tokens = self.estimate_messages_tokens(compressed_messages)
        remaining_tokens = available_for_messages - used_tokens

        history_messages = []
        for msg in reversed(messages):
            msg_tokens = self.estimate_message_tokens(msg)

            if msg_tokens > remaining_tokens:
                if msg.get("content"):
                    compressed_content = self.compress_large_data(
                        msg["content"], remaining_tokens - 100
                    )
                    compressed_msg = {**msg, "content": compressed_content}
                    compressed_tokens = self.estimate_message_tokens(compressed_msg)

                    if compressed_tokens <= remaining_tokens:
                        history_messages.insert(0, compressed_msg)
                        remaining_tokens -= compressed_tokens
                else:
                    continue
            else:
                history_messages.insert(0, msg)
                remaining_tokens -= msg_tokens

            if remaining_tokens < 500:
                break

        if last_user_msg and last_user_msg not in history_messages:
            if history_messages:
                history_messages[-1] = last_user_msg
            else:
                history_messages.append(last_user_msg)

        final_messages = compressed_messages + history_messages
        final_tokens = self.estimate_messages_tokens(final_messages)

        logger.info(
            f"Context compressed: {current_tokens} -> {final_tokens} tokens ({len(messages)} -> {len(history_messages)} messages)"
        )

        return final_messages
