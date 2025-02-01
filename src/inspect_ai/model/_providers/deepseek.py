import os
from typing import Any

from typing_extensions import override
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._providers.util import environment_prerequisite_error, model_base_url

from .openai import OpenAIAPI
from inspect_ai.model._chat_message import ChatMessageAssistant

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

        # Add timeout for slower models
        if "timeout" not in model_args:
            model_args["timeout"] = 60.0  # Default 1 minute timeout
            if "reasoner" in model_name.lower():
                model_args["timeout"] = 300.0  # 5 minutes for reasoner

        # Clean up model name - use just the model type for API
        if "/" in model_name:
            model_name = model_name.split("/")[1]  # Remove prefix
        if not any(prefix in model_name for prefix in ["deepseek-", "deepseek/"]):
            model_name = f"deepseek-{model_name}"  # Add prefix if missing

        # Configure reasoner model settings
        if "reasoner" in model_name.lower():
            # Remove unsupported parameters
            config.temperature = None
            config.top_p = None
            config.presence_penalty = None
            config.frequency_penalty = None
            config.logprobs = None
            config.top_logprobs = None
            
            # Set max tokens to handle long reasoning chains
            if "max_tokens" not in model_args:
                model_args["max_tokens"] = 4096  # Default max for final answer

        # Initialize through OpenAIAPI parent class
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
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

    @override
    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # For reasoner model, remove any reasoning_content from previous messages
        if "reasoner" in self.model_name.lower():
            return [{k: v for k, v in msg.items() if k != "reasoning_content"} 
                   for msg in messages]
        return messages 

    def _chat_choices_from_response(self, response: Any, tools: list[Any]) -> list[Any]:
        choices = super()._chat_choices_from_response(response, tools)
        
        # For reasoner model, add reasoning_content to the assistant message
        if "reasoner" in self.model_name.lower():
            for choice in choices:
                if isinstance(choice.message, ChatMessageAssistant):
                    # Get reasoning_content from response if available
                    if hasattr(response, "choices") and response.choices:
                        first_choice = response.choices[0]
                        if hasattr(first_choice, "reasoning_content"):
                            choice.message.reasoning_content = first_choice.reasoning_content

        return choices 