import os
from typing import Any

from typing_extensions import override
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._providers.util import environment_prerequisite_error, model_base_url

from .openai import OpenAIAPI

DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"


class DeepSeekAPI(OpenAIAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        # Resolve base URL
        base_url = model_base_url(base_url, "DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"

        # Resolve API key
        api_key = api_key or os.environ.get(DEEPSEEK_API_KEY)
        if not api_key:
            raise environment_prerequisite_error("DeepSeek", [DEEPSEEK_API_KEY])

        # Initialize through OpenAIAPI parent class
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
            **model_args
        ) 