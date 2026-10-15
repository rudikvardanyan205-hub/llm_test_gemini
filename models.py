"""Request and result schemas, with validation before network calls."""
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PromptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1)
    model: str = Field(default="gemini-2.5-flash", min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator("prompt", "model")
    @classmethod
    def cannot_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace")
        return value


class LLMResult(BaseModel):
    model: str
    response_text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    estimated_cost: float = Field(ge=0)
