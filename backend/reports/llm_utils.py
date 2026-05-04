from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .schemas import validate_model


def load_reports_api_key() -> str | None:
    env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()

    key_path = Path(__file__).resolve().with_name("api_key")
    if not key_path.exists():
        return None
    key = key_path.read_text(encoding="utf-8").strip()
    return key or None


def get_reports_llm():
    api_key = load_reports_api_key()
    if api_key and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = api_key

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("REPORT_LLM_MODEL", "gemini-2.5-flash"),
        temperature=float(os.getenv("REPORT_LLM_TEMPERATURE", "0.2")),
    )


def run_llm_json_chain(
    system_prompt: str,
    human_template: str,
    variables: dict[str, Any],
    output_model: type[BaseModel] | None = None,
    max_retries: int = 2,
) -> dict[str, Any]:
    from langchain.prompts import PromptTemplate

    llm = get_reports_llm()
    last_error: Exception | None = None
    for attempt in range(max(1, max_retries + 1)):
        retry_hint = ""
        if last_error is not None:
            retry_hint = (
                "\n\nThe previous response failed JSON validation. "
                f"Error: {last_error}. Return corrected JSON only."
            )
        prompt = PromptTemplate(
            input_variables=list(variables.keys()),
            template=f"{system_prompt.strip()}\n\n{human_template.strip()}{retry_hint}",
        )
        chain = prompt | llm
        response = chain.invoke(variables)
        raw = getattr(response, "content", response)
        try:
            parsed = extract_json_object(raw)
            if output_model is not None:
                return validate_model(output_model, parsed)
            return parsed
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
    raise RuntimeError(f"LLM JSON chain failed: {last_error}")


def extract_json_object(raw: Any) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain a JSON object")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON was not an object")
    return parsed


def dumps_for_prompt(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
