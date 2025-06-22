from langfuse_handler import ExperimentRunner, DatasetGenerator
import json
import asyncio

# dataset_generator = DatasetGenerator()

# with open('data/attestation_test_cases_50_v2.json', 'r') as f:
#     data = json.load(f)

# dataset_generator.generate_dataset('attestation-test-cases', data=data, sleep_delay=2.5)

experiment_runner = ExperimentRunner()
experiment_runner.run_experiment('attestation-test2', 'attestation-schema', 'attestation-test-cases', evaluator=None, experiment_params={}, test_count=None, requests_per_minute=100)

run = asyncio.run(experiment_runner.get_dataset_run('attestation-test-cases', 'attestation-test', limit=100, requests_per_minute=25, trace_info=['input', 'output'], dataset_item_info=['expected_output']))

with open('data/attestation_test.json', 'w') as f:
    json.dump(run, f, indent=4)