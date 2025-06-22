from langfuse_handler import PromptGenerator
import json

with open('data/attestation-prompt.json', 'r') as f:
    prompt = json.load(f)

prompt_generator = PromptGenerator()

prompt_generator.generate_prompt('attestation-schema', prompt, type='chat',
                                  config={
                                            "model": "gpt-4.1-mini",
                                            "temperature": 0,
                                            "json_schema": {
                                                "skill": "skill title or undefined",
                                                "bot_response": "bot's response to attestation"
                                            }
                                        }
                                  )