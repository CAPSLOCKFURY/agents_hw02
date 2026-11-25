"""
Модуль Human-in-the-Loop (HITL) для ризикових дій у Plan-and-Execute графі (Завдання 4).

Перед виконанням інструмента з множини `RISKY_TOOLS` (наприклад, `book_hotel`)
вузол `executor_with_hitl` викликає `langgraph.types.interrupt()`, зупиняє виконання графа,
зберігає стан у `SqliteSaver` і чекає на рішення людини через `Command(resume=...)`:
- approve: виконати інструмент із запропонованими параметрами;
- reject:  скасувати дію та передати причину відмови до стану агента;
- edit:    змінити параметри інструмента та виконати його з оновленими аргументами.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.types import interrupt

from config import get_candidate_llms, get_llm, invoke_with_fallback
from knowledge import search_knowledge
from plan_execute import (
    PlanExecuteState,
    _extract_response_text,
    _extract_user_query,
    build_plan_execute_graph,
)
from tools import ALL_DOMAIN_TOOLS, RISKY_TOOLS


def make_executor_with_hitl(tools: list[BaseTool], llm=None):
    """Створює вузол executor із Human-in-the-Loop approval gate через `interrupt()`."""
    model = llm or get_llm()
    tool_llms = [m.bind_tools(tools) for m in get_candidate_llms(model)]
    tools_by_name = {t.name: t for t in tools}

    def executor_with_hitl(state: PlanExecuteState) -> dict:
        step_idx = state["current_step"]
        plan = state["plan"]

        if step_idx >= len(plan):
            return {"completed": True}

        current_step = plan[step_idx]
        prev_results = "\n".join(state["results"]) if state["results"] else "Поки немає."

        prompt = (
            "Ти — виконавець кроків туристичного AI-асистента з контролем ризикових дій.\n"
            f"Початковий запит користувача: {_extract_user_query(state)}\n"
            f"Головна ціль: {state.get('goal') or _extract_user_query(state)}\n"
            f"Поточний крок ({step_idx + 1}/{len(plan)}): {current_step}\n\n"
            f"Результати попередніх кроків:\n{prev_results}\n\n"
            "Виконай ТІЛЬКИ поточний крок, викликавши відповідний інструмент. "
            "Використовуй точні числа, назви й дати з початкового запиту користувача та попередніх результатів. "
            "Якщо поточний крок передбачає бронювання готелю — обов'язково виклич інструмент `book_hotel` "
            "з точними параметрами (назва готелю, місто, дата check_in у форматі YYYY-MM-DD, кількість ночей, "
            "загальна вартість total_cost_eur)."
        )

        response = invoke_with_fallback(tool_llms, prompt)
        executed_tools: list[str] = []
        outputs: list[str] = []

        if getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = dict(tc["args"])
                tool_fn = tools_by_name.get(tool_name)

                if tool_fn is None:
                    outputs.append(f"{tool_name}: інструмент не знайдено")
                    continue

                if tool_name in RISKY_TOOLS:
                    # ── HITL Approval Gate: зупиняємо граф і чекаємо рішення людини ──
                    decision = interrupt(
                        {
                            "action": tool_name,
                            "args": tool_args,
                            "step_index": step_idx + 1,
                            "step_description": current_step,
                            "message": (
                                f"ПІДТВЕРДІТЬ РИЗИКОВУ ДІЮ (Approval Gate):\n"
                                f"  Крок {step_idx + 1}: {current_step}\n"
                                f"  Інструмент: {tool_name}\n"
                                f"  Параметри: {tool_args}\n"
                                f"  Очікувана відповідь: approve / reject / edit"
                            ),
                        }
                    )

                    # Нормалізуємо формат відповіді (підтримуємо як action=approve/reject/edit, так і approved=True/False)
                    if isinstance(decision, dict):
                        action = decision.get("action")
                        if action is None:
                            if decision.get("edited_args"):
                                action = "edit"
                            elif decision.get("approved") is True:
                                action = "approve"
                            else:
                                action = "reject"
                    else:
                        action = "approve" if bool(decision) else "reject"

                    if action == "edit":
                        edited_patch = decision.get("args") or decision.get("edited_args") or {}
                        final_args = {**tool_args, **edited_patch}
                        executed_tools.append(f"{tool_name}:edited")
                        try:
                            tool_result = tool_fn.invoke(final_args)
                            outputs.append(
                                f"[HITL EDIT] Оператор змінив параметри "
                                f"({tool_args} -> {final_args}). {tool_result}"
                            )
                        except Exception as exc:
                            outputs.append(
                                f"[HITL EDIT ERROR] Помилка валідації змінених параметрів {final_args}: {exc}"
                            )

                    elif action == "approve":
                        executed_tools.append(f"{tool_name}:approved")
                        try:
                            tool_result = tool_fn.invoke(tool_args)
                            outputs.append(f"[HITL APPROVED] {tool_result}")
                        except Exception as exc:
                            outputs.append(f"[HITL ERROR] Помилка виконання {tool_name}: {exc}")

                    else:
                        reason = (
                            decision.get("reason", "Без зазначення причини")
                            if isinstance(decision, dict)
                            else "Відхилено оператором"
                        )
                        executed_tools.append(f"{tool_name}:rejected")
                        outputs.append(
                            f"[HITL REJECTED] Дію `{tool_name}` відхилено оператором. "
                            f"Причина: {reason}. Бронювання НЕ виконано."
                        )
                else:
                    # Безпечний інструмент виконується автоматично без зупинки
                    executed_tools.append(tool_name)
                    try:
                        tool_result = tool_fn.invoke(tool_args)
                        outputs.append(f"{tool_name}({tool_args}) -> {tool_result}")
                    except Exception as exc:
                        outputs.append(f"{tool_name}({tool_args}) -> ПОМИЛКА: {exc}")

            step_result = "\n".join(outputs)
        else:
            step_result = _extract_response_text(response) or "Крок виконано."

        formatted_entry = f"Крок {step_idx + 1} [{current_step}]: {step_result}"
        return {
            "current_step": step_idx + 1,
            "results": [*state["results"], formatted_entry],
            "tools_used": executed_tools,
            "messages": [AIMessage(content=f"[Executor-HITL] {formatted_entry}")],
        }

    return executor_with_hitl


def build_hitl_graph(
    checkpointer,
    tools: list[BaseTool] | None = None,
    llm=None,
    max_replans: int = 2,
):
    """
    Збирає Plan-and-Execute граф із HITL approval gate та обов'язковим checkpointer.
    """
    if checkpointer is None:
        raise ValueError(
            "Для Human-in-the-Loop (interrupt/resume) обов'язково потрібно передати checkpointer (SqliteSaver)."
        )

    model = llm or get_llm()
    active_tools = tools if tools is not None else [*ALL_DOMAIN_TOOLS, search_knowledge]
    hitl_executor = make_executor_with_hitl(active_tools, llm=model)

    return build_plan_execute_graph(
        tools=active_tools,
        llm=model,
        checkpointer=checkpointer,
        executor_fn=hitl_executor,
        max_replans=max_replans,
    )
