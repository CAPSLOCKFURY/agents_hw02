"""
Інструменти (tools) для туристичного Plan-and-Execute агента з Pydantic-схемами та валідацією.

Містить:
- Безпечні інструменти: calculate, get_weather, convert_currency, search_hotels
- Ризиковий інструмент (потребує HITL-підтвердження): book_hotel
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator


# ── Безпечний AST-обчислювач математичних виразів ────────────────────────────
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast_node(node: ast.AST) -> float:
    """Рекурсивно обчислює лише дозволені арифметичні вузли AST."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Занадто великий показник степеня.")
        return _SAFE_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_eval_ast_node(node.operand))
    raise ValueError(f"Непідтримувана операція у виразі: {ast.dump(node)}")


# ── Pydantic-схеми вхідних аргументів ────────────────────────────────────────

class CalculateInput(BaseModel):
    """Схема валідації для математичних обчислень."""

    expression: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Арифметичний вираз для обчислення, наприклад '(120 * 4) + 35 + 80'.",
    )

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, value: str) -> str:
        cleaned = value.strip()
        if not re.fullmatch(r"[0-9+\-*/().%\s]+", cleaned):
            raise ValueError(
                "Вираз може містити лише цифри, пробіли, дужки та оператори + - * / %."
            )
        return cleaned


class WeatherInput(BaseModel):
    """Схема валідації для прогнозу погоди."""

    city: str = Field(
        ...,
        min_length=2,
        max_length=60,
        description="Назва європейського міста (наприклад: Париж, Рим, Барселона, Прага, Відень).",
    )
    month: str = Field(
        default="травень",
        description="Місяць подорожі українською або англійською (наприклад: травень, жовтень, січень).",
    )


class CurrencyConvertInput(BaseModel):
    """Схема валідації для конвертації валют."""

    amount: float = Field(
        ...,
        gt=0,
        le=1_000_000,
        description="Сума для конвертації (має бути більшою за 0).",
    )
    from_currency: Literal["EUR", "USD", "UAH", "GBP", "PLN"] = Field(
        ...,
        description="Вихідна валюта: EUR, USD, UAH, GBP або PLN.",
    )
    to_currency: Literal["EUR", "USD", "UAH", "GBP", "PLN"] = Field(
        ...,
        description="Цільова валюта: EUR, USD, UAH, GBP або PLN.",
    )


class HotelSearchInput(BaseModel):
    """Схема валідації для пошуку готелів."""

    city: str = Field(
        ...,
        min_length=2,
        max_length=60,
        description="Місто пошуку готелю (наприклад: Париж, Рим, Барселона, Прага).",
    )
    max_price_per_night_eur: float = Field(
        default=200.0,
        gt=0,
        le=2000.0,
        description="Максимальна прийнятна ціна за одну ніч у EUR.",
    )
    min_stars: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Мінімальна категорія готелю у зірках (від 1 до 5).",
    )


class BookHotelInput(BaseModel):
    """Схема валідації для ризикового інструмента бронювання готелю."""

    hotel_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Точна назва готелю для бронювання.",
    )
    city: str = Field(
        ...,
        min_length=2,
        max_length=60,
        description="Місто розташування готелю.",
    )
    check_in: str = Field(
        ...,
        description="Дата заїзду у форматі YYYY-MM-DD (наприклад: 2026-05-15).",
    )
    nights: int = Field(
        ...,
        ge=1,
        le=30,
        description="Кількість ночей проживання (від 1 до 30).",
    )
    total_cost_eur: float = Field(
        ...,
        gt=0,
        le=50_000,
        description="Загальна вартість бронювання в євро (EUR).",
    )

    @field_validator("check_in")
    @classmethod
    def validate_date_format(cls, value: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
            raise ValueError("Дата check_in повинна мати формат YYYY-MM-DD.")
        return value.strip()


# ── Дані для детермінованої демонстрації ─────────────────────────────────────

_WEATHER_DATA: dict[str, dict[str, str]] = {
    "париж": {
        "весна": "+15..+21°C, мінлива хмарність, можливий короткий дощ; легка куртка та парасоля.",
        "літо": "+22..+29°C, сонячно, іноді спекотно; легкий одяг та сонцезахисні окуляри.",
        "осінь": "+12..+18°C, прохолодні ранки, періодичні дощі; тренч або демісезонна куртка.",
        "зима": "+2..+8°C, сиро та вітряно, дощі; тепле пальто та шарф.",
    },
    "рим": {
        "весна": "+18..+24°C, сонячно й комфортно для піших прогулянок.",
        "літо": "+28..+35°C, спекотно й сухо; обов'язково головний убір та вода.",
        "осінь": "+16..+23°C, теплий оксамитовий сезон, помірні опади.",
        "зима": "+6..+13°C, м'яка зима, сонячні дні чергуються з дощами.",
    },
    "барселона": {
        "весна": "+17..+22°C, приємний морський бриз, переважно ясно.",
        "літо": "+25..+31°C, сонячно, висока вологість біля моря.",
        "осінь": "+17..+23°C, тепло й комфортно, у жовтні можливі короткі зливи.",
        "зима": "+9..+15°C, сонячно й м'яко, без снігу.",
    },
    "прага": {
        "весна": "+12..+19°C, свіжо, квітучі парки, можливі дощі.",
        "літо": "+20..+27°C, тепло, іноді грози.",
        "осінь": "+9..+16°C, прохолодно, часті тумани вранці.",
        "зима": "-2..+4°C, холодно, можливий сніг; теплий зимовий одяг.",
    },
    "відень": {
        "весна": "+13..+20°C, комфортно, вітряно на відкритих площах.",
        "літо": "+22..+29°C, тепло та сонячно.",
        "осінь": "+10..+17°C, помірна прохолода, ідеально для музеїв.",
        "зима": "-1..+4°C, прохолодно, святкові ярмарки у грудні.",
    },
}

_CITY_ALIASES = {
    "paris": "париж",
    "париж": "париж",
    "парижі": "париж",
    "rome": "рим",
    "рим": "рим",
    "римі": "рим",
    "barcelona": "барселона",
    "барселона": "барселона",
    "барселоні": "барселона",
    "prague": "прага",
    "прага": "прага",
    "празі": "прага",
    "vienna": "відень",
    "відень": "відень",
    "відні": "відень",
}

_MONTH_TO_SEASON = {
    "грудень": "зима", "січень": "зима", "лютий": "зима",
    "december": "зима", "january": "зима", "february": "зима",
    "березень": "весна", "квітень": "весна", "травень": "весна",
    "march": "весна", "april": "весна", "may": "весна",
    "червень": "літо", "липень": "літо", "серпень": "літо",
    "june": "літо", "july": "літо", "august": "літо",
    "вересень": "осінь", "жовтень": "осінь", "листопад": "осінь",
    "september": "осінь", "october": "осінь", "november": "осінь",
    "весна": "весна", "літо": "літо", "осінь": "осінь", "зима": "зима",
}

# Курси валют відносно EUR
_RATES_TO_EUR: dict[str, float] = {
    "EUR": 1.0,
    "USD": 0.92,
    "UAH": 0.0222,  # ~45.05 UAH за 1 EUR
    "GBP": 1.17,
    "PLN": 0.233,
}

_HOTELS_CATALOG: dict[str, list[dict]] = {
    "париж": [
        {"name": "Generator Paris Hostel", "stars": 2, "price_eur": 42, "district": "10-й округ"},
        {"name": "Hotel Ibis Paris Bastille", "stars": 3, "price_eur": 115, "district": "Бастилія"},
        {"name": "Hotel Eiffel Seine", "stars": 3, "price_eur": 145, "district": "біля Ейфелевої вежі"},
        {"name": "Le Marais Boutique Hotel", "stars": 4, "price_eur": 220, "district": "Маре"},
    ],
    "рим": [
        {"name": "YellowSquare Rome Hostel", "stars": 2, "price_eur": 38, "district": "Терміні"},
        {"name": "Hotel Artemide Express", "stars": 3, "price_eur": 110, "district": "Віа Націонале"},
        {"name": "Colosseo Garden Hotel", "stars": 4, "price_eur": 185, "district": "Монті"},
    ],
    "барселона": [
        {"name": "Sant Jordi Hostel Rock Palace", "stars": 2, "price_eur": 40, "district": "Ешампле"},
        {"name": "Hotel Jazz Barcelona", "stars": 3, "price_eur": 125, "district": "площа Каталонії"},
        {"name": "Catalonia Eixample 1864", "stars": 4, "price_eur": 195, "district": "Ешампле"},
    ],
    "прага": [
        {"name": "Czech Inn Hostel", "stars": 2, "price_eur": 28, "district": "Виногради"},
        {"name": "Hotel Archibald City", "stars": 4, "price_eur": 95, "district": "Нове Місто"},
    ],
}

# Реєстр фактично виконаних бронювань (дозволяє програмно перевіряти HITL approve/reject/edit)
BOOKING_LEDGER: list[dict] = []


# ── Реалізація інструментів ──────────────────────────────────────────────────

@tool(args_schema=CalculateInput)
def calculate(expression: str) -> str:
    """Обчислює математичний вираз (підрахунок бюджету, суми, вартості ночей тощо).

    Використовуйте для будь-яких арифметичних розрахунків.
    НЕ використовуйте для пошуку довідкової інформації.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_ast_node(tree.body)
        formatted = int(value) if value.is_integer() else round(value, 2)
        return f"Результат обчислення `{expression}` = {formatted}"
    except Exception as exc:
        return f"Помилка обчислення виразу `{expression}`: {exc}"


@tool(args_schema=WeatherInput)
def get_weather(city: str, month: str = "травень") -> str:
    """Повертає прогноз погоди та рекомендації щодо одягу для європейського міста у вказаний місяць.

    Використовуйте для перевірки погодних умов у конкретному місті.
    """
    city_key = _CITY_ALIASES.get(city.strip().lower(), city.strip().lower())
    month_key = month.strip().lower()
    season = _MONTH_TO_SEASON.get(month_key, "весна")

    city_weather = _WEATHER_DATA.get(city_key)
    if not city_weather:
        return (
            f"Для міста '{city}' немає локального прогнозу; "
            f"орієнтовна погода для Центральної Європи ({season}): +12..+20°C, мінлива хмарність."
        )
    return f"Погода у місті {city.capitalize()} ({month}, сезон: {season}): {city_weather[season]}"


@tool(args_schema=CurrencyConvertInput)
def convert_currency(
    amount: float,
    from_currency: Literal["EUR", "USD", "UAH", "GBP", "PLN"],
    to_currency: Literal["EUR", "USD", "UAH", "GBP", "PLN"],
) -> str:
    """Конвертує грошову суму між валютами EUR, USD, UAH, GBP та PLN.

    Використовуйте, коли потрібно перевести бюджет або ціну в іншу валюту.
    """
    amount_in_eur = amount * _RATES_TO_EUR[from_currency]
    converted = amount_in_eur / _RATES_TO_EUR[to_currency]
    return f"{amount:.2f} {from_currency} = {converted:.2f} {to_currency}"


@tool(args_schema=HotelSearchInput)
def search_hotels(
    city: str,
    max_price_per_night_eur: float = 200.0,
    min_stars: int = 1,
) -> str:
    """Шукає доступні готелі та хостели в місті з урахуванням максимальної ціни за ніч і зірковості.

    Використовуйте перед бронюванням, щоб знайти конкретні варіанти проживання та їхні ціни.
    """
    city_key = _CITY_ALIASES.get(city.strip().lower(), city.strip().lower())
    hotels = _HOTELS_CATALOG.get(city_key, [])
    if not hotels:
        return f"Готелі у місті '{city}' не знайдені в каталозі."

    matched = [
        h for h in hotels
        if h["price_eur"] <= max_price_per_night_eur and h["stars"] >= min_stars
    ]
    if not matched:
        cheapest = min(hotels, key=lambda h: h["price_eur"])
        return (
            f"НЕ ЗНАЙДЕНО готелів у місті {city.capitalize()} з критеріями: "
            f"до {max_price_per_night_eur} EUR/ніч та від {min_stars}*. "
            f"Найдешевший доступний варіант у місті: {cheapest['name']} "
            f"({cheapest['stars']}*, {cheapest['price_eur']} EUR/ніч, район: {cheapest['district']})."
        )

    lines = [
        f"- {h['name']} ({h['stars']}*, {h['price_eur']} EUR/ніч, район: {h['district']})"
        for h in matched
    ]
    return f"Знайдені варіанти проживання у місті {city.capitalize()}:\n" + "\n".join(lines)


@tool(args_schema=BookHotelInput)
def book_hotel(
    hotel_name: str,
    city: str,
    check_in: str,
    nights: int,
    total_cost_eur: float,
) -> str:
    """Бронює готель із фінансовим списанням коштів (РИЗИКОВА ДІЯ — потребує підтвердження людини!).

    Викликайте цей інструмент тільки тоді, коли користувач просить забронювати готель.
    """
    booking_id = f"BK-{len(BOOKING_LEDGER) + 1001}"
    record = {
        "booking_id": booking_id,
        "hotel_name": hotel_name,
        "city": city,
        "check_in": check_in,
        "nights": nights,
        "total_cost_eur": total_cost_eur,
    }
    BOOKING_LEDGER.append(record)
    return (
        f"БРОНЮВАННЯ ПІДТВЕРДЖЕНО [{booking_id}]: {hotel_name} ({city}), "
        f"заїзд {check_in}, {nights} ночей, сума {total_cost_eur:.2f} EUR."
    )


# Списки та словники інструментів
SAFE_TOOLS = [calculate, get_weather, convert_currency, search_hotels]
RISKY_TOOLS = {"book_hotel"}
ALL_DOMAIN_TOOLS = SAFE_TOOLS + [book_hotel]
TOOLS_BY_NAME = {t.name: t for t in ALL_DOMAIN_TOOLS}
