import dotenv
import os
from langfuse.openai import OpenAI
from langfuse import get_client
import ast

"""
langfuse_handler.py
-------------------

Utilities for managing, generating, and running prompts with Langfuse and LLMs (e.g., OpenAI).

This module provides classes to create prompts in Langfuse and to run prompts using a specified model client.
Environment variables are loaded from a .env file specified by the user.

Classes:
    PromptGenerator: For creating prompts in Langfuse.
    PromptRunner: For retrieving and running prompts using a model client.

Environment Variables Required:
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
    LANGFUSE_HOST
    OPENAI_API_KEY (if using OpenAI)
"""

class PromptGenerator:
    """
    Facilitates the creation of prompts in Langfuse.

    Args:
        env_path (str): Path to the .env file containing required environment variables.
    """
    def __init__(self, env_path: str = None):
        """
        Initialize PromptGenerator and load environment variables.

        Args:
            env_path (str): Path to the .env file. Defaults to ".env".
        """
        if env_path:
            dotenv.load_dotenv(env_path)
            os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
            os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
            os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")
        self.langfuse = get_client()

    def generate_prompt(self, prompt_name: str, prompt: str | list, **kwargs) -> str:
        """
        Create a prompt in Langfuse.

        Args:
            prompt_name (str): The name of the prompt.
            prompt (str | list): The prompt content to create. If a list is provided, it will be joined into a string.

        Returns:
            None
        """
        self.langfuse.create_prompt(
            name=prompt_name,
            prompt=prompt,
            **kwargs
        )

class PromptRunner:
    """
    Retrieves and runs prompts from Langfuse using a specified model client (default: OpenAI).

    Args:
        prompt_name (str): Name of the prompt to retrieve and run.
        env_path (str): Path to the .env file. Defaults to ".env".
        model_client: Model client to use (default: OpenAI).
    """
    def __init__(self, prompt_name: str, env_path: str = None, model_client: str = "openai", **kwargs):
        """
        Initialize PromptRunner, load environment variables, and retrieve the prompt.

        Args:
            prompt_name (str): Name of the prompt to retrieve and run.
            env_path (str): Path to the .env file. Defaults to ".env".
            model_client: Model client to use (default: OpenAI).
        """
        if env_path:
            dotenv.load_dotenv(env_path)
            os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
            os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
            os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
            
        if model_client == "openai":
            self.model_client = OpenAI()
        else:
            self.model_client = model_client
        self.langfuse = get_client()

        self.prompt = self.langfuse.get_prompt(prompt_name, **kwargs)


    def run_prompt(self, input_data: dict, config: dict = {}) -> str:
        """
        Run the specified prompt with provided input data and configuration.

        Args:
            input_data (dict): Input data for the prompt.
            config (dict, optional): Configuration for the model client. If not provided, uses the prompt's default config.

        Returns:
            str: The generated response from the model client.
        """
        if not config:
            config = self.prompt.config

        # stringify the json schema if it exists
        json_schema = config.pop('json_schema', {})
        if json_schema:
            input_data['json_schema_str'] = ', '.join([f"'{key}': {value}" for key, value in json_schema.items()])
            
        messages = self.prompt.compile(
            **input_data
        )
        
        response = self.model_client.chat.completions.create(
            messages=messages,
            **config
        )
        return response.choices[0].message.content
    
    def get_prompt(self):
        prompt_list = self.prompt.get_langchain_prompt()
        prompt_json = [{"role": role, "content": content} for role, content in prompt_list]
        return prompt_json

