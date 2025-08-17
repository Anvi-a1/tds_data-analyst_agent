import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import json
import time

# Set environment variables for testing
import os

os.environ["LLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "test_key"
os.environ["USE_SANDBOX"] = "false"

from app.orchestrator import Orchestrator, TimeoutException


@pytest.fixture
def orchestrator():
    """Fixture to create an Orchestrator instance for testing."""
    with patch("app.providers.ProviderFactory.create_provider") as mock_create_provider:
        mock_provider = MagicMock()
        mock_provider.provider_name = "mock_openai"
        mock_provider.generate_response.return_value = {
            "content": '{"result": "final answer"}'
        }
        mock_create_provider.return_value = mock_provider
        yield Orchestrator(), mock_provider


@pytest.mark.asyncio
async def test_handle_request_no_tool_calls(orchestrator):
    """
    Test that the orchestrator can handle a simple request without tool calls.
    """
    orch, mock_provider = orchestrator
    question = "What is the capital of France?"

    result = await orch.handle_request(question, time.time(), 60)

    assert result == {"result": "final answer"}
    mock_provider.generate_response.assert_called_once()


@pytest.mark.asyncio
@patch(
    "app.orchestrator.run_python_with_uv",
    return_value={"output": "hello", "success": True},
)
async def test_handle_request_with_tool_call(mock_run_python, orchestrator):
    """
    Test that the orchestrator can handle a request that requires a tool call.
    """
    orch, mock_provider = orchestrator

    function_call_response = {
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "run_python_with_uv",
                    "arguments": json.dumps({"code": "print('hello')"}),
                },
            }
        ]
    }
    final_response = {"content": '{"result": "tool output processed"}'}
    mock_provider.generate_response.side_effect = [
        function_call_response,
        final_response,
    ]

    question = "Run a python script to print hello"
    result = await orch.handle_request(question, time.time(), 60)

    assert result == {"result": "tool output processed"}
    assert mock_provider.generate_response.call_count == 2
    mock_run_python.assert_called_once()


@pytest.mark.asyncio
async def test_handle_request_timeout(orchestrator):
    """
    Test that the orchestrator handles a timeout correctly.
    """
    orch, mock_provider = orchestrator

    with pytest.raises(TimeoutException):
        await orch.handle_request("some question", time.time() - 70, 60)


@pytest.mark.asyncio
@patch("app.orchestrator.run_python_with_uv", side_effect=Exception("Execution failed"))
async def test_handle_request_tool_call_error(mock_run_python, orchestrator):
    """
    Test how the orchestrator handles an error during a tool call.
    """
    orch, mock_provider = orchestrator

    function_call_response = {
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "run_python_with_uv",
                    "arguments": json.dumps({"code": "invalid code"}),
                },
            }
        ]
    }
    final_response = {"content": '{"error": "Something went wrong"}'}
    mock_provider.generate_response.side_effect = [
        function_call_response,
        final_response,
    ]

    question = "Run some python code"
    result = await orch.handle_request(question, time.time(), 60)

    assert "error" in result
    mock_run_python.assert_called_once()

    messages = mock_provider.generate_response.call_args[0][0]
    last_message = messages[-1]
    assert last_message["role"] == "tool"
    assert "error" in last_message["content"]
    assert "Execution failed" in last_message["content"]
