import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from client import estimate_cost, send_prompt
from config import MissingAPIKeyError, Settings
from main import load_prompts, run_requests, save_results
from models import PromptRequest


def test_validation():
    with pytest.raises(ValidationError):
        PromptRequest(prompt=" ")
    with pytest.raises(ValidationError):
        PromptRequest(prompt="Hi", temperature=3)


def test_missing_key():
    with pytest.raises(MissingAPIKeyError):
        Settings(_env_file=None, gemini_api_key=None).require_api_key()


def test_json_validation(tmp_path: Path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps([{"prompt": "ok"}, {"prompt": ""}, "nice", 42]))
    good, bad = load_prompts(path, "gemini-2.5-flash")
    assert len(good) == 2 and len(bad) == 2


def test_cost():
    assert estimate_cost("gemini-2.5-flash", 1_000_000, 1_000_000) == pytest.approx(2.8)


class FakeModels:
    async def generate_content(self, **kwargs):
        await asyncio.sleep(0)
        if kwargs["contents"] == "bad":
            raise ValueError("synthetic failure")
        return SimpleNamespace(
            usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=50),
            text="test response",
        )


@pytest.mark.asyncio
async def test_async_batch(tmp_path: Path):
    fake = SimpleNamespace(models=FakeModels())
    good = PromptRequest(prompt="good")
    result = await send_prompt(fake, good)
    assert result.response_text == "test response"
    results, errors = await run_requests(fake, [good, PromptRequest(prompt="bad"), good], 2)
    assert len(results) == 2 and len(errors) == 1
    path = tmp_path / "results.json"
    save_results(path, results)
    assert len(json.loads(path.read_text())) == 2
