import os
import time
from typing import Dict, List

from tqdm import tqdm

from src.data_loader import build_prompt_for_short_answer
from src.utils import load_jsonl, save_jsonl


class BaseLLMClient:
    """
    Base class for LLM clients.
    """

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class DummyLLMClient(BaseLLMClient):
    """
    Dummy client for testing the pipeline without calling an API.
    """

    def __init__(self, fixed_response: str = ""):
        self.fixed_response = fixed_response

    def generate(self, prompt: str) -> str:
        return self.fixed_response


class OpenAICompatibleLLMClient(BaseLLMClient):
    """
    OpenAI-compatible API client.

    Works for:
    - OpenAI
    - Qwen / DashScope compatible mode
    - other OpenAI-compatible providers
    """

    def __init__(
        self,
        model: str,
        api_key_env: str,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 64,
        sleep_seconds: float = 0.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is not installed. Run: pip install openai"
            ) from exc

        api_key = os.getenv(api_key_env)

        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_env} is not set. "
                f"Please run: export {api_key_env}='your_api_key'"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.sleep_seconds = sleep_seconds

    def generate(self, prompt: str) -> str:
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer science questions concisely and directly. "
                        "Return only the short answer, without explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return response.choices[0].message.content.strip()


def create_llm_client(config: Dict) -> BaseLLMClient:
    """
    Create an LLM client from config.

    Supported providers:
    - dummy
    - openai
    - qwen
    """
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "dummy")

    if provider == "dummy":
        return DummyLLMClient(
            fixed_response=llm_config.get("fixed_response", "")
        )

    if provider == "openai":
        return OpenAICompatibleLLMClient(
            model=llm_config.get("model", "gpt-4o-mini"),
            api_key_env=llm_config.get("api_key_env", "OPENAI_API_KEY"),
            base_url=llm_config.get("base_url", None),
            temperature=llm_config.get("temperature", 0.0),
            max_tokens=llm_config.get("max_tokens", 64),
            sleep_seconds=llm_config.get("sleep_seconds", 0.0),
        )

    if provider == "qwen":
        return OpenAICompatibleLLMClient(
            model=llm_config.get("model", "qwen3.5-plus"),
            api_key_env=llm_config.get("api_key_env", "DASHSCOPE_API_KEY"),
            base_url=llm_config.get(
                "base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            temperature=llm_config.get("temperature", 0.0),
            max_tokens=llm_config.get("max_tokens", 64),
            sleep_seconds=llm_config.get("sleep_seconds", 0.0),
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_predictions_for_records(
    records: List[Dict],
    llm_client: BaseLLMClient,
    use_support: bool = False,
) -> List[Dict]:
    """
    Generate predictions for a list of QA records.

    Args:
        records: Records containing question and ground_truth.
        llm_client: LLM client.
        use_support: Whether to include support context in the prompt.

    Returns:
        New records with prompt and prediction fields.
    """
    output_records = []

    for record in tqdm(records, desc="Generating predictions"):
        question = record["question"]

        if use_support:
            from src.data_loader import build_prompt_with_support

            prompt = build_prompt_with_support(
                question=question,
                support=record.get("support", ""),
            )
        else:
            prompt = build_prompt_for_short_answer(question)

        try:
            prediction = llm_client.generate(prompt)
            error_message = ""
        except Exception as exc:
            prediction = ""
            error_message = str(exc)

        new_record = dict(record)
        new_record["prompt"] = prompt
        new_record["prediction"] = prediction

        if error_message:
            new_record["generation_error"] = error_message

        output_records.append(new_record)

    return output_records


def generate_predictions_from_file(
    input_path: str,
    output_path: str,
    llm_client: BaseLLMClient,
    use_support: bool = False,
    sample_size: int | None = None,
) -> None:
    """
    Load records from JSONL, generate predictions, save to JSONL.
    """
    records = load_jsonl(input_path)

    # Teacher-processed data uses correct_answer.
    # Convert it to the project format expected by later scripts.
    normalized_records = []
    for idx, record in enumerate(records):
        new_record = dict(record)

        if "id" not in new_record:
            new_record["id"] = f"sample_{idx}"

        if "ground_truth" not in new_record and "correct_answer" in new_record:
            new_record["ground_truth"] = new_record["correct_answer"]

        normalized_records.append(new_record)

    records = normalized_records

    if sample_size is not None:
        records = records[:sample_size]
        print(f"Using first {sample_size} records.")

    output_records = generate_predictions_for_records(
        records=records,
        llm_client=llm_client,
        use_support=use_support,
    )

    save_jsonl(output_records, output_path)