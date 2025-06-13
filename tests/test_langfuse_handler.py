import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Ensure src/ is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from langfuse_handler.langfuse_handler import PromptGenerator, PromptRunner

@patch('langfuse_handler.langfuse_handler.Langfuse')
def test_prompt_generator_init(mock_langfuse):
    gen = PromptGenerator(env_path=".env")
    assert gen.env_path == ".env"
    assert hasattr(gen, 'langfuse')
    mock_langfuse.assert_called_once()

@patch('langfuse_handler.langfuse_handler.Langfuse')
def test_prompt_generator_generate_prompt(mock_langfuse):
    mock_instance = mock_langfuse.return_value
    gen = PromptGenerator(env_path=".env")
    gen.generate_prompt("test", "prompt", "chat", {"model": "gpt-3.5-turbo"})
    mock_instance.create_prompt.assert_called_once_with(
        name="test",
        prompt="prompt",
        type="chat",
        config={"model": "gpt-3.5-turbo"}
    )

@patch('langfuse_handler.langfuse_handler.Langfuse')
@patch('langfuse_handler.langfuse_handler.OpenAI')
def test_prompt_runner_init(mock_openai, mock_langfuse):
    runner = PromptRunner("test-prompt", env_path=".env")
    assert runner.env_path == ".env"
    assert hasattr(runner, 'langfuse')
    assert hasattr(runner, 'model_client')
    mock_openai.assert_called_once()
    mock_langfuse.assert_called_once()

@patch('langfuse_handler.langfuse_handler.Langfuse')
@patch('langfuse_handler.langfuse_handler.OpenAI')
def test_prompt_runner_run_prompt(mock_openai, mock_langfuse):
    # Mock prompt and model_client
    mock_prompt = MagicMock()
    mock_prompt.config = {"model": "gpt-3.5-turbo"}
    mock_prompt.compile.return_value = [{"role": "user", "content": "hi"}]
    mock_langfuse.return_value.get_prompt.return_value = mock_prompt
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="response"))]
    mock_openai.return_value.chat.completions.create.return_value = mock_response

    runner = PromptRunner("test-prompt", env_path="../.env")
    result = runner.run_prompt({"foo": "bar"})
    assert result == "response"
    mock_prompt.compile.assert_called()
    mock_openai.return_value.chat.completions.create.assert_called() 