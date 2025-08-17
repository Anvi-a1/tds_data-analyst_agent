import os
from typing import Dict, Any
from dotenv import load_dotenv
from .base_provider import BaseLLMProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .openrouter_provider import OpenRouterProvider

load_dotenv()


class ProviderFactory:
    """Factory class to create and manage LLM providers"""

    _providers = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "openrouter": OpenRouterProvider,
    }

    @classmethod
    def create_provider(
        cls, provider_name: str = None, config: Dict[str, Any] = None
    ) -> BaseLLMProvider:
        """Create and initialize a provider"""
        if provider_name is None:
            provider_name = os.getenv("LLM_PROVIDER", "openai")

        if config is None:
            config = {}

        provider_name = provider_name.lower()

        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unsupported LLM provider: {provider_name}. Available: {available}"
            )

        provider_class = cls._providers[provider_name]
        provider = provider_class(config)
        provider.initialize()

        return provider

    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available providers"""
        return list(cls._providers.keys())

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """Register a new provider"""
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError("Provider class must inherit from BaseLLMProvider")

        cls._providers[name.lower()] = provider_class
