from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with configuration"""
        self.config = config
        self.logger = logger
        self.token_manager = None

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider client"""
        pass

    @abstractmethod
    def generate_response(self, messages: List[Dict], functions: List[Dict]) -> Dict:
        """Generate a response from the LLM"""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name"""
        pass

    def validate_config(self) -> bool:
        """Validate the provider configuration"""
        return True

    def log_info(self, message: str) -> None:
        """Log an info message with provider context"""
        self.logger.info(f"[{self.provider_name}] {message}")

    def log_error(self, message: str) -> None:
        """Log an error message with provider context"""
        self.logger.error(f"[{self.provider_name}] {message}")

    def log_warning(self, message: str) -> None:
        """Log a warning message with provider context"""
        self.logger.warning(f"[{self.provider_name}] {message}")

    def log_debug(self, message: str) -> None:
        """Log a debug message with provider context"""
        self.logger.debug(f"[{self.provider_name}] {message}")

    def _ensure_token_manager(self) -> None:
        """Initialize token manager if not already initialized"""
        if self.token_manager is None:
            from app.tools.token_manager import TokenManager

            model_name = getattr(self, "model", None) or getattr(
                self, "model_name", None
            )
            self.token_manager = TokenManager(model_name)

    def manage_context_size(
        self, messages: List[Dict], functions: List[Dict] = None
    ) -> List[Dict]:
        """Manage message context to fit within token limits"""
        self._ensure_token_manager()
        return self.token_manager.manage_conversation_context(messages, functions)
