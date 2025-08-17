# LLM Providers

This directory contains the modular LLM provider system for the Data Analyst Agent. Each provider is implemented as a separate class that inherits from `BaseLLMProvider`.

## Available Providers

### OpenAI Provider
- **Provider Name**: `openai`
- **Environment Variables**:
  - `OPENAI_API_KEY` (required)
  - `OPENAI_BASE_URL` (optional, defaults to `https://api.openai.com/v1`)
  - `OPENAI_MODEL` (optional, defaults to `gpt-4-0613`)

### Gemini Provider
- **Provider Name**: `gemini`
- **Environment Variables**:
  - `GEMINI_API_KEY` (required)
- **Requirements**: `google-genai` package

### OpenRouter Provider
- **Provider Name**: `openrouter`
- **Environment Variables**:
  - `OPENROUTER_API_KEY` (required)
  - `OPENROUTER_MODEL` (optional, defaults to `openai/gpt-4-turbo`)

## Usage

### Basic Usage
```python
from app.providers import ProviderFactory

# Create provider using environment variables
provider = ProviderFactory.create_provider("openai")

# Or with custom config
config = {
    "api_key": "your_api_key",
    "model": "gpt-4-turbo"
}
provider = ProviderFactory.create_provider("openai", config)

# Generate response
messages = [{"role": "user", "content": "Hello"}]
functions = []
response = provider.generate_response(messages, functions)
```

### Environment Configuration
Set the `LLM_PROVIDER` environment variable to choose which provider to use:

```bash
# Use OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_key

# Use Gemini
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=your_key

# Use OpenRouter
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=your_key
```

## Adding New Providers

To add a new provider:

1. Create a new file in the `providers` directory (e.g., `new_provider.py`)
2. Implement a class that inherits from `BaseLLMProvider`
3. Implement the required abstract methods:
   - `initialize()`
   - `generate_response()`
   - `provider_name` property
4. Register the provider in the factory:

```python
from .new_provider import NewProvider

# In factory.py
_providers = {
    # ... existing providers
    "newprovider": NewProvider,
}
```

## Architecture

- **BaseLLMProvider**: Abstract base class defining the interface
- **ProviderFactory**: Factory class for creating and managing providers
- **Individual Providers**: Concrete implementations for each LLM service

Each provider handles its own configuration, initialization, and response generation while maintaining a consistent interface.
