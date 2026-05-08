from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .schemas import validate_model


API_PROVIDERS = {"api", "gemini"}
LOCAL_PROVIDERS = {"local", "ollama", "qwen"}


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def normalize_llm_provider(provider: str | None = None) -> str:
    selected = (provider or os.getenv("REPORT_LLM_PROVIDER") or "").strip().lower()
    if not selected:
        selected = "local" if _env_flag("REPORT_LLM_USE_OLLAMA", "0") else "api"
    if selected in LOCAL_PROVIDERS:
        return "local"
    if selected in API_PROVIDERS:
        return "api"
    raise ValueError("llm_provider must be one of: api, local")


def get_ollama_model(model: str | None = None) -> str:
    return model or os.getenv("REPORT_OLLAMA_MODEL") or os.getenv("REPORT_LLM_MODEL", "qwen2.5:7b")


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def get_gemini_model(model: str | None = None) -> str:
    return model or os.getenv("REPORT_GEMINI_MODEL") or os.getenv("REPORT_LLM_MODEL", "gemini-2.5-flash")


def llm_runtime_info(provider: str | None = None, model: str | None = None) -> dict[str, Any]:
    selected_provider = normalize_llm_provider(provider)
    if selected_provider == "local":
        return {
            "provider": "local",
            "model": get_ollama_model(model),
            "base_url": get_ollama_base_url(),
        }
    return {
        "provider": "api",
        "model": get_gemini_model(model),
    }


def check_local_runtime(model: str | None = None, timeout_sec: float = 5.0) -> dict[str, Any]:
    selected_model = get_ollama_model(model)
    base_url = get_ollama_base_url()
    result: dict[str, Any] = {
        "provider": "local",
        "model": selected_model,
        "base_url": base_url,
        "server_ok": False,
        "model_available": False,
        "available_models": [],
    }
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        result["error"] = f"Ollama server is not reachable: {exc.reason}"
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result

    models = payload.get("models") if isinstance(payload, dict) else []
    names = [
        str(item.get("name") or item.get("model"))
        for item in models
        if isinstance(item, dict) and (item.get("name") or item.get("model"))
    ]
    result["server_ok"] = True
    result["available_models"] = names
    result["model_available"] = selected_model in names
    if not result["model_available"]:
        result["error"] = f"Model not found. Run: ollama pull {selected_model}"
    return result


def check_qwen_runtime(model: str | None = None, timeout_sec: float = 5.0) -> dict[str, Any]:
    return check_local_runtime(model=model, timeout_sec=timeout_sec)


def load_reports_api_key() -> str | None:
    env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()

    key_path = Path(__file__).resolve().with_name("api_key")
    if not key_path.exists():
        return None
    key = key_path.read_text(encoding="utf-8").strip()
    return key or None


def get_reports_llm(provider: str | None = None, model: str | None = None):
    selected_provider = normalize_llm_provider(provider)
    temperature = float(os.getenv("REPORT_LLM_TEMPERATURE", "0.2"))

    if selected_provider == "local":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=get_ollama_model(model),
            base_url=get_ollama_base_url(),
            temperature=temperature,
            format="json",
            num_ctx=int(os.getenv("REPORT_OLLAMA_NUM_CTX", "4096")),
            num_gpu=int(os.getenv("REPORT_OLLAMA_NUM_GPU", "-1")),
            keep_alive=os.getenv("REPORT_OLLAMA_KEEP_ALIVE", "5m"),
            sync_client_kwargs={"timeout": float(os.getenv("REPORT_OLLAMA_TIMEOUT_SEC", "120"))},
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = load_reports_api_key()
    if api_key and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = api_key

    return ChatGoogleGenerativeAI(
        model=get_gemini_model(model),
        temperature=temperature,
    )


def run_llm_json_chain(
    system_prompt: str,
    human_template: str,
    variables: dict[str, Any],
    output_model: type[BaseModel] | None = None,
    max_retries: int = 2,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    from langchain_core.prompts import PromptTemplate

    llm = get_reports_llm(provider=provider, model=model)
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
