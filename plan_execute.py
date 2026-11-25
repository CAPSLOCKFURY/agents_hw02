"""
Реалізація архітектури Plan-and-Execute у LangGraph (Завдання 1 та 2).

Вузли графа:
    START -> planner -> executor -> replanner -> (executor | END)

- Planner генерує структурований план (`Plan`) через `with_structured_output`.
- Executor виконує поточний крок плану за допомогою підключених інструментів.
- Replanner оцінює результати та повертає `ReplanDecision` (`continue` / `replan` / `finish`).
"""

from __future__ import annotations

import operator
from typing import Annotated, Callable, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import get_candidate_llms, get_llm, invoke_with_fallback


# ── Structured Output Pydantic-моделі ────────────────────────────────────────

class Plan(BaseModel):
    """Структурований план виконання задачі користувача."""

    goal: str = Field(
        description="Стисле формулювання головної цілі користувача."
    )
    steps: list[str] = Field(
        min_length=1,
        max_length=6,
        description=(
            "Упорядкований список із 2–5 конкретних, послідовних і виконуваних кроків "
            "для досягнення цілі."
        ),
    )


class ReplanDecision(BaseModel):
    """Рішення вузла replanner після виконання поточного кроку."""

    action: Literal["continue", "replan", "finish"] = Field(
        description=(
            "'continue' — продовжити виконання наступного кроку чинного плану; "
            "'replan' — змінити залишкові кроки плану через нові обставини, помилку або відмову; "
            "'finish' — завершити виконання, оскільки ціль повністю досягнута або подальші кроки недоцільні."
        )
    )
    updated_steps: list[str] | None = Field(
        default=None,
        description=(
            "Новий список ТІЛЬКИ залишкових кроків (не повторювати вже виконані). "
            "Заповнюється лише коли action == 'replan'."
        ),
    )
    reasoning: str = Field(
        description="Коротке обґрунтування прийнятого рішення українською мовою."
    )
    final_answer: str | None = Field(
        default=None,
        description=(
            "Підсумкова структурована відповідь користувачу українською мовою на основі всіх "
            "зібраних результатів. Заповнюється, коли action == 'finish'."
        ),
    )


# ── Стан LangGraph ───────────────────────────────────────────────────────────

class PlanExecuteState(TypedDict):
    """Стан графа Plan-and-Execute, що зберігається у checkpointer після кожного вузла."""

    messages: Annotated[list, operator.add]
    goal: str
    plan: list[str]
    current_step: int
    results: list[str]
    completed: bool
    replan_count: int
    tools_used: Annotated[list[str], operator.add]


def create_initial_state(user_query: str) -> PlanExecuteState:
    """Формує початковий стан графа для нового запиту користувача."""
    return {
        "messages": [HumanMessage(content=user_query)],
        "goal": "",
        "plan": [],
        "current_step": 0,
        "results": [],
        "completed": False,
        "replan_count": 0,
        "tools_used": [],
    }


def _extract_user_query(state: PlanExecuteState) -> str:
    """Дістає початковий текст запиту користувача з історії повідомлень."""
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    if state.get("messages"):
        first = state["messages"][0]
        return str(getattr(first, "content", first))
    return ""


def _describe_tools(tools: list[BaseTool]) -> str:
    """Формує короткий каталог доступних інструментів для промпту planner/replanner."""
    lines = []
    for t in tools:
        first_line = (t.description or "").strip().splitlines()[0]
        lines.append(f"- {t.name}: {first_line}")
    return "\n".join(lines)


def _extract_response_text(response) -> str:
    """Безпечно витягує текстовий вміст із відповіді ChatGoogleGenerativeAI."""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ]
        return "\n".join(p for p in parts if p).strip()
    return str(content)


# ── Фабрика вузлів та збірка графа ───────────────────────────────────────────

def make_planner_node(llm, tools: list[BaseTool]):
    """Створює вузол planner із structured output (`Plan`)."""
    planner_llms = [m.with_structured_output(Plan) for m in get_candidate_llms(llm)]
    tools_catalog = _describe_tools(tools)

    def planner_node(state: PlanExecuteState) -> dict:
        user_query = _extract_user_query(state)
        prompt = (
            "Ти — планувальник туристичного AI-асистента.\n"
            f"Запит користувача: {user_query}\n\n"
            f"Доступні інструменти:\n{tools_catalog}\n\n"
            "Склади чіткий план із 2–5 конкретних, послідовних кроків українською мовою.\n"
            "Правила планування:\n"
            "1. Кожен крок має бути атомарною дією, яку можна виконати через один інструмент "
            "(або фінальний підсумок).\n"
            "2. Для довідкових правил, віз, музеїв, багажу, страхування використовуй базу знань (search_knowledge).\n"
            "3. Для розрахунків використовуй calculate, для валют — convert_currency, для погоди — get_weather.\n"
            "4. Якщо користувач просить забронювати готель — додай окремий крок бронювання (book_hotel) "
            "після пошуку або підрахунку вартості.\n"
            "5. НЕ додавай крок бронювання, якщо користувач про нього не просив."
        )
        plan: Plan = invoke_with_fallback(planner_llms, prompt)
        steps_formatted = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan.steps))
        return {
            "goal": plan.goal,
            "plan": plan.steps,
            "current_step": 0,
            "results": [],
            "completed": False,
            "replan_count": 0,
            "messages": [
                AIMessage(content=f"[Planner] Ціль: {plan.goal}\nПлан:\n{steps_formatted}")
            ],
        }

    return planner_node


def make_executor_node(llm, tools: list[BaseTool]):
    """Створює стандартний вузол executor (без HITL) для виконання поточного кроку."""
    tool_llms = [m.bind_tools(tools) for m in get_candidate_llms(llm)]
    tools_by_name = {t.name: t for t in tools}

    def executor_node(state: PlanExecuteState) -> dict:
        step_idx = state["current_step"]
        plan = state["plan"]

        if step_idx >= len(plan):
            return {"completed": True}

        current_step = plan[step_idx]
        prev_results = "\n".join(state["results"]) if state["results"] else "Поки немає."

        prompt = (
            "Ти — виконавець кроків туристичного AI-асистента.\n"
            f"Початковий запит користувача: {_extract_user_query(state)}\n"
            f"Головна ціль: {state.get('goal') or _extract_user_query(state)}\n"
            f"Поточний крок ({step_idx + 1}/{len(plan)}): {current_step}\n\n"
            f"Результати попередніх кроків:\n{prev_results}\n\n"
            "Виконай ТІЛЬКИ поточний крок. Обов'язково виклич інструмент, який відповідає поточному кроку "
            "(наприклад, calculate для обчислень, convert_currency для валют, get_weather для погоди, "
            "search_knowledge для довідки, search_hotels для пошуку готелів). "
            "Використовуй точні числа, дати й назви з початкового запиту користувача та попередніх результатів."
        )

        response = invoke_with_fallback(tool_llms, prompt)
        executed_tools: list[str] = []
        outputs: list[str] = []

        if getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_fn = tools_by_name.get(tool_name)
                executed_tools.append(tool_name)

                if tool_fn is None:
                    outputs.append(f"{tool_name}: інструмент не знайдено")
                    continue

                try:
                    tool_result = tool_fn.invoke(tool_args)
                    outputs.append(f"{tool_name}({tool_args}) -> {tool_result}")
                except Exception as exc:
                    outputs.append(f"{tool_name}({tool_args}) -> ПОМИЛКА валідації/виконання: {exc}")

            step_result = "\n".join(outputs)
        else:
            step_result = _extract_response_text(response) or "Крок виконано без додаткового виводу."

        formatted_entry = f"Крок {step_idx + 1} [{current_step}]: {step_result}"
        return {
            "current_step": step_idx + 1,
            "results": [*state["results"], formatted_entry],
            "tools_used": executed_tools,
            "messages": [AIMessage(content=f"[Executor] {formatted_entry}")],
        }

    return executor_node


def make_replanner_node(llm, tools: list[BaseTool], max_replans: int = 2):
    """Створює вузол replanner із structured output (`ReplanDecision`)."""
    replanner_llms = [m.with_structured_output(ReplanDecision) for m in get_candidate_llms(llm)]
    tools_catalog = _describe_tools(tools)

    def replanner_node(state: PlanExecuteState) -> dict:
        plan = state["plan"]
        step_idx = state["current_step"]
        results = state["results"]
        replan_count = state.get("replan_count", 0)
        remaining = plan[step_idx:]
        user_query = _extract_user_query(state)

        results_text = "\n".join(results) if results else "Немає."
        remaining_text = (
            "\n".join(f"  - {s}" for s in remaining) if remaining else "Залишкових кроків немає."
        )

        prompt = (
            "Ти — репланувальник (Replanner) туристичного Plan-and-Execute агента.\n"
            f"Початковий запит користувача: {user_query}\n"
            f"Ціль: {state.get('goal', user_query)}\n"
            f"Поточний план ({len(plan)} кроків): {plan}\n"
            f"Виконано кроків: {step_idx}/{len(plan)}\n"
            f"Кількість попередніх перепланувань: {replan_count} (максимум: {max_replans})\n\n"
            f"Отримані результати:\n{results_text}\n\n"
            f"Залишкові кроки:\n{remaining_text}\n\n"
            f"Доступні інструменти:\n{tools_catalog}\n\n"
            "Прийми рішення (continue / replan / finish):\n"
            "- Обери 'continue', якщо залишкові кроки актуальні й потрібні для повної відповіді на запит.\n"
            "- Обери 'replan' і задай `updated_steps` (ТІЛЬКИ нові залишкові кроки), якщо попередній крок "
            "повернув 'НЕ ЗНАЙДЕНО', помилку або несподіване обмеження, через яке потрібно скоригувати план "
            "(наприклад, пом'якшити критерії пошуку готелю або перевірити альтернативний варіант).\n"
            "- Обери 'finish' і сформуй якісний `final_answer` українською мовою, якщо всі частини запиту "
            "вже виконано, або якщо оператор відхилив ризикову дію (бронювання) і продовжувати не потрібно.\n"
            "- НЕ обирай 'finish', якщо ще є важливі невиконані кроки запиту користувача."
        )

        decision: ReplanDecision = invoke_with_fallback(replanner_llms, prompt)

        # Захист від нескінченних перепланувань або некоректного continue після останнього кроку
        if decision.action == "replan" and decision.updated_steps and replan_count < max_replans:
            new_plan = [*plan[:step_idx], *decision.updated_steps]
            return {
                "plan": new_plan,
                "current_step": step_idx,
                "replan_count": replan_count + 1,
                "messages": [
                    AIMessage(
                        content=(
                            f"[Replanner -> REPLAN #{replan_count + 1}] {decision.reasoning}\n"
                            f"Оновлені залишкові кроки: {decision.updated_steps}"
                        )
                    )
                ],
            }

        if decision.action == "continue" and step_idx < len(plan):
            return {
                "messages": [
                    AIMessage(content=f"[Replanner -> CONTINUE] {decision.reasoning}")
                ]
            }

        # Завершення (action == 'finish' або кроки вичерпано)
        final_text = decision.final_answer or (
            f"{decision.reasoning}\n\nПідсумок виконаних кроків:\n" + "\n".join(results)
        )
        return {
            "completed": True,
            "messages": [
                AIMessage(content=f"[Replanner -> FINISH] {decision.reasoning}"),
                AIMessage(content=final_text),
            ],
        }

    return replanner_node


def should_end(state: PlanExecuteState) -> Literal["executor", "__end__"]:
    """Умовний маршрутизатор після replanner: завершення або наступний крок executor."""
    if state.get("completed") or state.get("current_step", 0) >= len(state.get("plan", [])):
        return "__end__"
    return "executor"


def build_plan_execute_graph(
    tools: list[BaseTool],
    llm=None,
    checkpointer=None,
    executor_fn: Callable[[PlanExecuteState], dict] | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    max_replans: int = 2,
):
    """
    Збирає та компілює LangGraph StateGraph для Plan-and-Execute агента.

    Аргументи:
        tools: Список доступних інструментів агента.
        llm: Екземпляр LLM (якщо None — створюється через config.get_llm()).
        checkpointer: Checkpointer (наприклад, SqliteSaver) для persistence/HITL.
        executor_fn: Опціональний кастомний вузол executor (наприклад, з HITL).
        interrupt_before: Опціональний список вузлів для зупинки ДО виконання.
        interrupt_after: Опціональний список вузлів для зупинки ПІСЛЯ виконання.
        max_replans: Максимальна кількість перепланувань за один запуск.
    """
    model = llm or get_llm()

    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", make_planner_node(model, tools))
    graph.add_node("executor", executor_fn or make_executor_node(model, tools))
    graph.add_node("replanner", make_replanner_node(model, tools, max_replans=max_replans))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "replanner")
    graph.add_conditional_edges(
        "replanner",
        should_end,
        {"executor": "executor", "__end__": END},
    )

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )
