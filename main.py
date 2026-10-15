"""Command-line interface for single, batch, and model-comparison runs."""
import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from google import genai
from pydantic import ValidationError

from client import PromptExecutionError, UnknownModelPriceError, send_prompt
from config import MissingAPIKeyError, Settings
from models import LLMResult, PromptRequest


class PromptFileError(ValueError):
    """Raised for invalid file-level prompt JSON."""


def load_prompts(path: Path, default_model: str) -> tuple[list[PromptRequest], list[str]]:
    """Validate independently so one bad item does not discard the others."""
    try:
        contents: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromptFileError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if isinstance(contents, dict) and isinstance(contents.get("prompts"), list):
        contents = contents["prompts"]
    if not isinstance(contents, list):
        raise PromptFileError("Expected a JSON array or an object containing a 'prompts' array")
    valid: list[PromptRequest] = []
    errors: list[str] = []
    for number, entry in enumerate(contents, 1):
        try:
            if isinstance(entry, str):
                entry = {"prompt": entry}
            if not isinstance(entry, dict):
                raise ValueError("Expected a prompt object or nonempty string")
            valid.append(PromptRequest.model_validate({"model": default_model, **entry}))
        except (ValidationError, ValueError, TypeError) as exc:
            errors.append(f"Prompt {number}: {exc}")
    return valid, errors


def save_results(path: Path, results: list[LLMResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False), encoding="utf-8")


async def run_requests(client: object, requests: list[PromptRequest], limit: int, timeout: float = 30.0) -> tuple[list[LLMResult], list[str]]:
    semaphore = asyncio.Semaphore(limit)

    async def worker(number: int, request: PromptRequest) -> tuple[int, LLMResult | Exception]:
        async with semaphore:
            try:
                return number, await send_prompt(client, request, timeout=timeout)
            except Exception as exc:
                return number, exc

    outcomes = await asyncio.gather(*(worker(i, request) for i, request in enumerate(requests, 1)))
    results: list[LLMResult] = []
    errors: list[str] = []
    for number, outcome in outcomes:
        if isinstance(outcome, Exception):
            errors.append(f"Prompt {number}: {outcome}")
        else:
            results.append(outcome)
    return results, errors


def display_result(result: LLMResult) -> None:
    print(f"\nModel: {result.model}\nText: {result.response_text}")
    print(f"Tokens: {result.input_tokens} in / {result.output_tokens} out")
    print(f"Latency: {result.latency_ms:.0f} ms | Est. cost: ${result.estimated_cost:.6f}")


def display_comparison(results: list[LLMResult]) -> None:
    print(f"{'Model':<19} {'Chars':>7} {'Input':>7} {'Output':>7} {'Latency ms':>12} {'Cost USD':>12}")
    for r in results:
        print(f"{r.model:<19} {len(r.response_text):>7} {r.input_tokens:>7} {r.output_tokens:>7} {r.latency_ms:>12.0f} {r.estimated_cost:>12.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and compare LLM prompts with the Google GenAI async SDK")
    parser.add_argument("--model", help="Model for single prompts and default for file items")
    parser.add_argument("--file", type=Path, help="Input JSON file with prompts")
    parser.add_argument("--compare", nargs="+", metavar="MODEL", help="Compare a prompt across 2 or 3 model names")
    parser.add_argument("--prompt", help="Prompt text (otherwise input() in interactive modes)")
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = Settings()
    if args.compare and not 2 <= len(args.compare) <= 3:
        raise ValueError("--compare requires 2 or 3 model names")
    if args.compare and args.file:
        raise ValueError("Use --compare or --file, not both")
    model = args.model or settings.default_model
    key = settings.require_api_key()  # Fail before opening the network client.
    async with genai.Client(api_key=key).aio as client:
        if args.file:
            requests, errors = load_prompts(args.file, model)
        else:
            prompt = args.prompt if args.prompt is not None else input("Enter prompt: ")
            models = args.compare or [model]
            requests = [PromptRequest(prompt=prompt, model=name) for name in models]
            errors = []
        results, runtime_errors = await run_requests(client, requests, settings.max_concurrent, settings.request_timeout)
        errors.extend(runtime_errors)
    for result in results:
        if not args.compare:
            display_result(result)
    if args.compare:
        display_comparison(results)
    for error in errors:
        print(f"ERROR: {error}")
    save_results(args.output, results)
    print(f"\nSaved {len(results)} result(s) to {args.output}; {len(errors)} failure(s).")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (MissingAPIKeyError, PromptFileError, ValidationError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
