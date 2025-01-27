import pytest
<<<<<<< HEAD

from inspect_ai.model._chat_message import ChatMessageUser
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._providers.goodfire import GoodfireAPI


@pytest.mark.asyncio
async def test_goodfire_api() -> None:
    """Test the Goodfire API provider."""
    model = GoodfireAPI(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",  # Using exact model name from SUPPORTED_MODELS
=======
from test_helpers.utils import skip_if_no_goodfire

from inspect_ai.model._chat_message import ChatMessageUser
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import get_model


@pytest.mark.asyncio
@skip_if_no_goodfire
async def test_goodfire_api() -> None:
    """Test the Goodfire API provider."""
    model = get_model(
        "goodfire/meta-llama/Meta-Llama-3.1-8B-Instruct",
>>>>>>> 894d343a1fb2c84624808f8650452060c331aa7b
        config=GenerateConfig(
            max_tokens=50,  # Match other tests
        ),
    )

    message = ChatMessageUser(content="What is 2+2?")
<<<<<<< HEAD
    response = await model.generate(input=[message], tools=[], tool_choice="none")
=======
    response = await model.generate(input=[message])
>>>>>>> 894d343a1fb2c84624808f8650452060c331aa7b
    assert len(response.completion) >= 1
