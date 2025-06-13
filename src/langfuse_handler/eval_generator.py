from .langfuse_handler import PromptRunner
from langfuse import get_client, observe
import dotenv
import os
from tqdm import tqdm
import requests
import time

"""
eval_generator.py
----------------

Provides utilities for generating datasets and running experiments with Langfuse and LLMs.

Classes:
    DatasetGenerator: For creating datasets and dataset items in Langfuse.
    ExperimentRunner: For running LLM experiments and logging results to Langfuse.

Environment Variables Required:
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
    LANGFUSE_HOST
    OPENAI_API_KEY (if using OpenAI)
"""

class DatasetGenerator:
    """
    Facilitates the creation of datasets and dataset items in Langfuse.

    Args:
        env_path (str): Path to the .env file containing required environment variables.
    """
    def __init__(self, env_path: str = None):
        """
        Initialize DatasetGenerator and load environment variables.

        Args:
            env_path (str): Path to the .env file. Defaults to ".env".
        """
        if env_path:
            dotenv.load_dotenv(env_path)
            os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
            os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
            os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")
        self.langfuse = get_client()

    def create_dataset_item(self, dataset_name: str, item: dict):
        """
        Create a dataset item in Langfuse.

        Args:
            dataset_name (str): The name of the dataset.
            item (dict): The item data to add to the dataset.
        """
        self.langfuse.create_dataset_item(
            dataset_name=dataset_name,
            **item
        )
    
    def generate_dataset(self, dataset_name: str, dataset_params: dict = {}, data: list[dict] = None):
        """
        Create a dataset in Langfuse and optionally add items to it.

        Args:
            dataset_name (str): The name of the dataset.
            dataset_params (dict, optional): Additional parameters for dataset creation.
            data (list[dict], optional): List of items to add to the dataset.
        """
        self.langfuse.create_dataset(
            name=dataset_name,
            **dataset_params
        )
        
        if data is None:
            return

        for item in data:
            self.create_dataset_item(dataset_name, item)

class ExperimentRunner:
    """
    Runs LLM experiments and logs results to Langfuse.

    Args:
        env_path (str): Path to the .env file containing required environment variables.
    """
    def __init__(self, env_path: str = None):
        """
        Initialize ExperimentRunner and load environment variables.

        Args:
            env_path (str): Path to the .env file. Defaults to ".env".
        """
        if env_path:
            dotenv.load_dotenv(env_path)
            os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
            os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
            os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
            
        self.langfuse = get_client()

    @observe()
    def run_prompt(self, prompt_runner: PromptRunner, input_data: dict) -> str:
        """
        Run a custom LLM application using the provided PromptRunner and input data.

        Args:
            prompt_runner (PromptRunner): The PromptRunner instance to use.
            input_data (dict): Input data for the prompt.

        Returns:
            str: The response from the LLM.
        """
        response = prompt_runner.run_prompt(input_data)
        return response


    def prompt_app(self, prompt_runner: PromptRunner, input_data: dict, item_id: str, run_name: str) -> str:
        """
        Run a prompt application, log the generation and trace in Langfuse, and return the response.

        Args:
            prompt_runner (PromptRunner): The PromptRunner instance to use.
            input_data (dict): Input data for the prompt.
            item_id: Identifier for the dataset item.
            run_name: Name of the experiment run.

        Returns:
            str: The response from the LLM.
        """
        with self.langfuse.start_as_current_generation(
            name=f"{prompt_runner.prompt.name}-llm-call",
            input=input_data,
            metadata={"item_id": item_id, "run": run_name}, # Example metadata for the generation
            model=prompt_runner.prompt.config['model']
        ) as generation:
            # Simulate LLM call
            response = self.run_prompt(prompt_runner, input_data)

            generation.update(output=response)
    
            # Update the trace with the input and output
            generation.update_trace(
                input=input_data,
                output=response,
            )
    
            return response

    def run_experiment(self, experiment_name: str, prompt_name: str, dataset_name: str, evaluator: callable = None, experiment_params: dict = {}, test_count: int = None, requests_per_minute: int = 1000):
        """
        Run an experiment by iterating over dataset items, running prompts, and optionally evaluating responses.

        Args:
            experiment_name (str): Name of the experiment run.
            prompt_name (str): Name of the prompt to use.
            dataset_name (str): Name of the dataset to use.
            evaluator (callable, optional): Function to evaluate responses.
            experiment_params (dict, optional): Additional parameters for the experiment run.
            test_count (int, optional): Number of items to test (if None, use all items).
            requests_per_minute (int, optional): Max requests per minute. Defaults to 1000.
        """
        
        
        prompt_runner = PromptRunner(prompt_name, env_path=self.env_path)

        dataset = self.langfuse.get_dataset(dataset_name)

        sleep_time = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        for item in tqdm(dataset.items[:test_count], desc="Running experiments", unit="item"):

            with item.run(
                run_name=experiment_name,
                **experiment_params
            ) as root_span:
                response = self.prompt_app(prompt_runner, item.input, item.id, experiment_name)


                if evaluator is not None:
                    evaluator(response, item, root_span)
            if sleep_time > 0:
                time.sleep(sleep_time)  # Sleep between requests

    def make_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self.make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(v) for v in obj]
        elif hasattr(obj, '__dict__'):
            return self.make_serializable(obj.__dict__)
        elif hasattr(obj, '__str__') and not isinstance(obj, (str, bytes)):
            return str(obj)
        else:
            return obj

    async def get_dataset_run(self, dataset_name: str, run_name: str, limit: int = 100, requests_per_minute: int = 25, trace_info: list[str] = ['input', 'output'], dataset_item_info: list[str] = []):
        url = f"{os.environ['LANGFUSE_HOST']}/api/public/dataset-run-items"
        dataset = self.langfuse.get_dataset(dataset_name)
        datasetId = dataset.id

        response = requests.get(
            url,
            auth=(os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"]),
            params={
                "datasetId": datasetId,
                "runName": run_name,
                "limit": limit,
                "response": "json"
            }
        )

        try:
            response.raise_for_status()  # Raises HTTPError for bad responses
        except requests.HTTPError as e:
            print(f"API request failed: {e} - {response.text}")
            return None  # or raise, or handle as appropriate

        run_items = response.json()['data']

        sleep_time = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        run_traces = []

        item_lookup = {item.id: item for item in dataset.items}
        for trace in tqdm(run_items, desc="Fetching traces", unit="trace"):
            trace = self.langfuse.api.trace.get(trace['traceId'])
            item = item_lookup[trace.metadata['dataset_item_id']]
            trace_dict = {attr: getattr(trace, attr, None) for attr in trace_info}
            item_dict = {attr: getattr(item, attr, None) for attr in dataset_item_info}
            combined_dict = {**trace_dict, **item_dict}
            run_traces.append(combined_dict)
            if sleep_time > 0:
                time.sleep(sleep_time)  # Sleep between requests

        serialized_run_traces = [self.make_serializable(trace) for trace in run_traces]
        return serialized_run_traces
        
        