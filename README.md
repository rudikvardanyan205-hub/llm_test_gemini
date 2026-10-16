# Mini LLM Client

A small Python project that sends prompts to Google Gemini and saves the responses.

## Script Work

The script can:

- send one prompt
- read prompts from JSON files
- send multiple prompts asynchronously
- compare Gemini models
- save results to `results.json`

Prompts are validated before being sent to the API.

## Libraries Used

The project uses a few main libraries:

- `google-genai` - official Google library used to send requests to Gemini models
- `pydantic` - used to validate prompt data and response models
- `pydantic-settings` - used to load configuration values from `.env`
- `asyncio` - used to run several API requests concurrently
- `json` - used to read prompt files and save results
- `time` - used to measure request latency

## Prompt Sending

Prompts are sent using the Gemini async client.

The script collects:

- response text
- input tokens
- output tokens
- latency
- estimated cost

## Async

For batch requests, `asyncio.gather()` is used so multiple prompts can run at the same time.

This is faster than waiting for each prompt one by one.

## Exception Handling

The script handles:

- missing API key
- invalid prompt data
- API errors
- timeouts
- unsupported models

Failed API requests can be retried up to 3 times.

If one prompt fails during a batch run, the other prompts can still continue.

## Results

Successful responses are saved to:

```text
results.json


Run
Install dependencies:

uv run python main.py