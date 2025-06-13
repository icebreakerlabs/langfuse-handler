from langfuse_handler import PromptRunner
import json

prompt_runner = PromptRunner("skill-matcher", env_path="../.env")


response = prompt_runner.run_prompt(input_data={
    "job_title": "Software Engineer",
    "job_functions" : ['software development'],
    "job_description": "Need a software engineer with 5 years of experience in Python and Django.",
})



print(response)

with open('../data/prompt.json', 'w') as f:
    json.dump(prompt_runner.get_prompt(), f, indent=2)
