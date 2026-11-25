"""
Конфігурація проєкту: завантаження змінних середовища з .env та ініціалізація Gemini LLM
із підтримкою автоматичного fallback та повторних спроб при тимчасових 503/504 помилках API.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Базові шляхи проєкту
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
DB_PATH = BASE_DIR / "agent_state.db"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Завантажуємо змінні середовища з файлу .env у корені проєкту
load_dotenv(dotenv_path=ENV_PATH)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

# Резервні моделі Gemini на випадок тимчасового перевантаження (503/504)
GEMINI_FALLBACK_MODELS: list[str] = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]

T = TypeVar("T")


def get_llm(model: str | None = None, timeout: int = 20) -> ChatGoogleGenerativeAI:
    """
    Повертає налаштований екземпляр ChatGoogleGenerativeAI.

    Модель задається через змінну GEMINI_MODEL у файлі .env
    (за замовчуванням gemini-flash-lite-latest).
    """
    if not GOOGLE_API_KEY:
        raise ValueError(
            "Не знайдено GOOGLE_API_KEY у змінних середовища. "
            "Створіть файл .env на основі .env.example та вкажіть ваш ключ Gemini."
        )

    return ChatGoogleGenerativeAI(
        model=model or GEMINI_MODEL,
        api_key=GOOGLE_API_KEY,
        max_retries=0,
        timeout=timeout,
    )


def get_candidate_llms(primary_llm: ChatGoogleGenerativeAI | None = None) -> list[ChatGoogleGenerativeAI]:
    """Повертає список екземплярів Gemini LLM: основний + резервні без дублікатів."""
    primary = primary_llm or get_llm()
    primary_name = getattr(primary, "model", GEMINI_MODEL).replace("models/", "")
    candidates = [primary]
    for fallback_name in GEMINI_FALLBACK_MODELS:
        if fallback_name != primary_name:
            candidates.append(get_llm(model=fallback_name))
    return candidates


def invoke_with_fallback(
    runnables: list,
    prompt: str,
    rounds: int = 6,
    base_delay: float = 3.0,
) -> T:
    """
    Викликає список LangChain runnable (основна модель + резервні) із повторними спробами
    при тимчасових мережевих/серверних помилках Gemini API (503 UNAVAILABLE, 504 DEADLINE_EXCEEDED).
    """
    last_error: Exception | None = None
    for attempt in range(rounds):
        for runnable in runnables:
            try:
                return runnable.invoke(prompt)
            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                is_transient = any(
                    code in err_str
                    for code in ("503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED", "429", "RESOURCE_EXHAUSTED")
                )
                if not is_transient:
                    raise
        if attempt < rounds - 1:
            time.sleep(base_delay * (attempt + 1))

    assert last_error is not None
    raise last_error
