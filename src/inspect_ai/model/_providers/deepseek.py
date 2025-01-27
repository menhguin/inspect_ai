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
        # Call super first with api_key_vars
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=[DEEPSEEK_API_KEY],  # Let parent class know about our env var
            config=config,
        )

        # Resolve base URL
        base_url = model_base_url(base_url, "DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"

        # Resolve API key if not already set by parent
        if not self.api_key:
            self.api_key = os.environ.get(DEEPSEEK_API_KEY)
            if not self.api_key:
                raise environment_prerequisite_error("DeepSeek", [DEEPSEEK_API_KEY])

        # Initialize through OpenAIAPI parent class again with resolved values
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=self.api_key,
            config=config,
            **model_args
        )

    @override
    def is_o1(self) -> bool:
        return False  # DeepSeek doesn't use o1 model architecture

    @override
    def is_o1_full(self) -> bool:
        return False

    @override
    def is_o1_mini(self) -> bool:
        return False

    @override
    def is_o1_preview(self) -> bool:
        return False 