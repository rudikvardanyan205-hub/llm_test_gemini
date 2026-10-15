"""Gemini async generation, retries, token accounting, and cost estimates."""
import asyncio
from time import perf_counter
from typing import Any

from google.genai import errors, types
from models import LLMResult, PromptRequest

PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.8-flash": (0.75, 3.75),
}

class PromptExecutionError(RuntimeError):
    """A provider request could not be completed."""


class UnknownModelPriceError(ValueError):
    """No price is configured for the requested model."""


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICES_PER_MILLION:
        raise UnknownModelPriceError(f"Unknown price for {model!r}; configure its current rates first.")
    input_price, output_price = PRICES_PER_MILLION[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


async def send_prompt(client: Any, request: PromptRequest, timeout: float = 30.0,
                      retries: int = 3) -> LLMResult:
    """Call Google GenAI asynchronously; retry only transient problems."""
    estimate_cost(request.model, 0, 0)
    started = perf_counter()
    for attempt in range(1, retries + 1):
        try:
            response = await asyncio.wait_for(
                client.models.generate_content(
                    model=request.model,
                    contents=request.prompt,
                    config=types.GenerateContentConfig(temperature=request.temperature),
                ), timeout=timeout,
            )
            usage = response.usage_metadata
            if usage is None:
                raise PromptExecutionError("Gemini did not provide token usage data")
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            return LLMResult(
                model=request.model,
                response_text=response.text or "",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=(perf_counter() - started) * 1000,
                estimated_cost=estimate_cost(request.model, input_tokens, output_tokens),
            )
        except (asyncio.TimeoutError, ConnectionError, OSError) as exc:
            if attempt == retries:
                raise PromptExecutionError(f"{request.model}: failed after {attempt} attempts: {exc}") from exc
        except errors.APIError as exc:
            code = exc.code or 0
            if code not in {408, 409, 429} and code < 500:
                raise PromptExecutionError(f"{request.model}: non-retryable API error {code}: {exc}") from exc
            if attempt == retries:
                raise PromptExecutionError(f"{request.model}: API failure after {attempt} attempts: {exc}") from exc
        except PromptExecutionError:
            raise
        if attempt < retries:
            await asyncio.sleep(min(2 ** (attempt - 1), 4))
    raise PromptExecutionError("No response was returned")
