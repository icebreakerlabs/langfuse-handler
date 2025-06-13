# langfuse-llm: Langfuse Handler Library

## Overview

The `langfuse-llm` library provides utilities and integrations for managing, generating, and running prompts with [Langfuse](https://langfuse.com/) and large language models (LLMs) such as OpenAI. It is designed to streamline prompt engineering, experiment tracking, and dataset generation for LLM-based applications.

## Features
- **Prompt Management:** Create and manage prompts in Langfuse.
- **Prompt Execution:** Run prompts using LLMs (e.g., OpenAI) with easy configuration.
- **Experiment Tracking:** Run and log experiments with prompt/LLM combinations.
- **Dataset Generation:** Create and manage datasets for evaluation and benchmarking.
- **Environment Management:** Loads environment variables from a user-specified `.env` file.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd langfuse-llm
   ```
2. Install dependencies (recommended: use a virtual environment):
   ```bash
   pip install -r requirements.txt
   # or, if using pyproject.toml
   pip install .
   ```

## Environment Setup

Create a `.env` file in your project root (or specify a custom path) with the following variables:

```
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=your_langfuse_host_url
OPENAI_API_KEY=your_openai_api_key  # Required if using OpenAI
```

## Project Structure

```
langfuse-llm/
├── src/
│   └── langfuse_handler/
│       ├── langfuse_handler.py      # Main prompt management and runner classes
│       └── eval_generator.py        # Dataset and experiment utilities
├── sample_usage/
│   └── promptrunner_test.py         # Example usage script
├── skill_matcher_eval/
│   ├── experimenter.py              # Example experiment runner
│   └── dataset.py                   # Example dataset generator
├── .env                             # Environment variables (not committed)
├── README.md                        # This file
└── ...
```

## Usage Examples

### Creating and Running a Prompt
```python
from src.langfuse_handler import PromptGenerator, PromptRunner

# Create a prompt in Langfuse
prompt_gen = PromptGenerator("../.env")
prompt_gen.generate_prompt(
    prompt_name="skill-matcher",
    prompt="What skills are required for the following job? {job_description}",
    type="chat",
    config={"model": "gpt-3.5-turbo"}
)

# Run a prompt
prompt_runner = PromptRunner("skill-matcher", "../.env")
response = prompt_runner.run_prompt(input_data={
    "job_title": "Software Engineer",
    "job_functions": ["software development"],
    "job_description": "Need a software engineer with 5 years of experience in Python and Django."
})
print(response)
```

### Running an Experiment
```python
from src.langfuse_handler import ExperimentRunner

experiment_runner = ExperimentRunner("../.env")
experiment_runner.run_experiment(
    experiment_name="skill-matcher-experiment",
    prompt_name="skill-matcher",
    dataset_name="icebreaker-jobs"
)
```

## Conventions
- Python 3.8+
- PEP8 code style
- All public classes and methods are documented with docstrings
- Use relative imports within the `src/langfuse_handler/` package

## License

This project is licensed under the MIT License.
