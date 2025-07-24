"""levelapp/core/evaluator.py"""

from typing import Dict

from levelapp.clients import AnthropicClient, MistralClient
from levelapp.core.base import BaseEvaluator, BaseChatClient

prompt = """
Your task is to evaluate how well the agent's generated text matches the expected text.
Use the following classification criteria:

3 - Excellent Match: The generated text is virtually identical to the expected text with no meaningful differences.
2 - Good Match: The generated text closely matches the expected text with only minor wording differences.
1 - Moderate Match: The generated text captures the main ideas but has noticeable differences or omissions.
0 - Poor Match: The generated text has significant differences and misses several key points.

Expected Output:
\"\"\"
{reference_text}
\"\"\"

Agent's Output:
\"\"\"
{generated_text}
\"\"\"

Return your evaluation as a valid JSON object with exactly these keys:
{{"match_level": <an integer between 1 and 5>, "justification": <a brief explanation>}}

Output only the JSON object and nothing else.
"""


class InteractionEvaluator(BaseEvaluator):
    def __init__(self):
        self.clients: Dict[str, BaseChatClient] = {}

    def register_client(self, provider: str, client: BaseChatClient):
        self.clients[provider] = client

    @staticmethod
    def build_prompt(generated_text: str, reference_text: str) -> str:
        return prompt.format(generated_text=generated_text, reference_text=reference_text)

    def evaluate(self, provider: str, generated_text: str, reference_text: str):
        if provider not in self.clients:
            raise ValueError(f"[InteractionEvaluator] The client {provider} is not registered.")

        prompt = self.build_prompt(generated_text=generated_text, reference_text=reference_text)
        client = self.clients[provider]
        response = client.call(message=prompt)

        print(f"[InteractionEvaluator] Evaluation result: {response}")

    async def async_evaluate(self, provider: str, generated_text: str, reference_text: str):
        if provider not in self.clients:
            raise ValueError(f"[InteractionEvaluator] The client {provider} is not registered.")

        prompt = self.build_prompt(generated_text=generated_text, reference_text=reference_text)
        client = self.clients[provider]
        response = await client.acall(message=prompt)

        print(f"[InteractionEvaluator] Evaluation result: {response}")


if __name__ == '__main__':
    from dotenv import load_dotenv

    from levelapp.clients.ionos import IonosClient
    from levelapp.clients.openai import OpenAIClient

    load_dotenv()

    evaluator = InteractionEvaluator()
    ionos_client = IonosClient(base_url="https://inference.de-txl.ionos.com/models")
    openai_client = OpenAIClient()
    anthropic_client = AnthropicClient()
    mistral_client = MistralClient()

    evaluator.register_client("ionos", openai_client)
    evaluator.register_client("openai", openai_client)
    evaluator.register_client("anthropic", anthropic_client)
    evaluator.register_client("mistral", mistral_client)

    reference = "dubito, ergo sum, vel, quod idem est, cogito, ergo sum"
    generated = "cogito, ergo sum"
    evaluator.evaluate("mistral", generated, reference)