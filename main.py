"""
Головний демонстраційний скрипт для ДЗ №2:
Plan-and-Execute агент у LangGraph з Persistence (SqliteSaver), Agentic RAG (ChromaDB) та HITL.

Запуск усіх демонстрацій:
    python main.py
Або окремого завдання:
    python main.py --task 1
    python main.py --task 2
    python main.py --task 3
    python main.py --task 4
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from config import DB_PATH, GEMINI_MODEL, get_llm
from hitl import build_hitl_graph
from knowledge import get_knowledge_collection, search_knowledge
from plan_execute import build_plan_execute_graph, create_initial_state
from tools import ALL_DOMAIN_TOOLS, BOOKING_LEDGER, SAFE_TOOLS

# Гарантуємо коректний вивід українського тексту у консолі Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _banner(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n  {title}\n{line}")


def _print_result_summary(result: dict) -> None:
    print(f"\nЦіль: {result.get('goal')}")
    print("План:")
    for idx, step in enumerate(result.get("plan", []), start=1):
        print(f"  {idx}. {step}")
    print(f"Виконано кроків: {result.get('current_step')}/{len(result.get('plan', []))}")
    print(f"Кількість перепланувань: {result.get('replan_count', 0)}")
    print(f"Використані інструменти: {result.get('tools_used', [])}")
    print("\nРезультати кроків:")
    for res in result.get("results", []):
        print(f"  • {res}")
    if result.get("messages"):
        print("\nФінальна відповідь агента:")
        print(result["messages"][-1].content)


# ── ЗАВДАННЯ 1: Plan-and-Execute у LangGraph ─────────────────────────────────

def demo_task1_plan_execute() -> None:
    _banner("ЗАВДАННЯ 1: Plan-and-Execute у LangGraph (3 сценарії різної складності)")

    llm = get_llm()
    tools = [*SAFE_TOOLS, search_knowledge]
    app = build_plan_execute_graph(tools=tools, llm=llm)

    scenarios = [
        (
            "Приклад 1 (Базовий, 2–3 кроки): Погода та бюджет",
            "Перевір погоду в Римі у травні та порахуй бюджет на 4 ночі по 110 EUR "
            "плюс 18 EUR квиток у Колізей, і переведи загальну суму в UAH.",
        ),
        (
            "Приклад 2 (Складний, 4–5 кроків): Комплексне планування подорожі",
            "Сплануй 3-денну поїздку до Парижа у квітні: перевір правила безвізу до ЄС, "
            "дізнайся графік роботи Лувру, знайди готель 3* до 150 EUR за ніч та порахуй "
            "загальну вартість проживання на 3 ночі в EUR і в UAH.",
        ),
        (
            "Приклад 3 (Динамічний Replanning): Нереалістичний бюджет готелю",
            "Знайди в Парижі готель категорії 4 зірки дешевше 60 EUR за ніч та порахуй вартість 2 ночей у ньому.",
        ),
    ]

    for title, query in scenarios:
        print(f"\n--- {title} ---")
        print(f"Запит: {query}")
        result = app.invoke(create_initial_state(query))
        _print_result_summary(result)


# ── ЗАВДАННЯ 2: Checkpointer та Persistence (SqliteSaver) ────────────────────

def demo_task2_persistence(reset_db: bool = True) -> None:
    _banner("ЗАВДАННЯ 2: Persistence та відновлення стану через SqliteSaver")

    if reset_db and DB_PATH.exists():
        DB_PATH.unlink()

    llm = get_llm()
    tools = [*SAFE_TOOLS, search_knowledge]
    config_session_1 = {"configurable": {"thread_id": "session-paris-001"}}

    # Фаза 1: Запускаємо граф із зупинкою після ПЕРШОГО кроку executor (імітація збою/перезапуску)
    print("\n[Фаза 1] Перший запуск `session-paris-001` із зупинкою посередині плану (після кроку 1)...")
    conn1 = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    try:
        saver1 = SqliteSaver(conn1)
        app_before_crash = build_plan_execute_graph(
            tools=tools,
            llm=llm,
            checkpointer=saver1,
            interrupt_after=["executor"],
        )
        query1 = (
            "Перевір погоду в Парижі у вересні, знайди готель до 150 EUR/ніч "
            "та порахуй вартість 3 ночей у найдешевшому 3* готелі."
        )
        app_before_crash.invoke(create_initial_state(query1), config=config_session_1)
        paused_state = app_before_crash.get_state(config_session_1)
        print(f"  Зупинено після кроку: {paused_state.values['current_step']}/{len(paused_state.values['plan'])}")
        print(f"  Наступний вузол у черзі: {paused_state.next}")
        print(f"  Збережений план: {paused_state.values['plan']}")
        print(f"  Збережені проміжні результати: {paused_state.values['results']}")
    finally:
        conn1.close()
        print(f"  [!] З'єднання з {DB_PATH.name} закрито (імітація перезапуску процесу).")

    # Фаза 2: Нове з'єднання з agent_state.db та новий скомпільований граф — продовжуємо той самий thread_id
    print("\n[Фаза 2] Перезапуск: відкриваємо `agent_state.db` заново та відновлюємо `session-paris-001`...")
    conn2 = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    try:
        saver2 = SqliteSaver(conn2)
        app_after_restart = build_plan_execute_graph(
            tools=tools,
            llm=llm,
            checkpointer=saver2,
        )
        restored_state = app_after_restart.get_state(config_session_1)
        print(
            f"  Відновлено з SQLite -> step={restored_state.values['current_step']}, "
            f"next={restored_state.next}, completed={restored_state.values['completed']}"
        )

        # Продовжуємо виконання з checkpoint (передаємо None як вхід!)
        final_session_1 = app_after_restart.invoke(None, config=config_session_1)
        print("  Сесію `session-paris-001` успішно доведено до завершення після відновлення:")
        _print_result_summary(final_session_1)

        # Фаза 3: Незалежна сесія з іншим thread_id у тій самій БД
        print("\n[Фаза 3] Запуск незалежної сесії `session-rome-002` у тому ж файлі SQLite...")
        config_session_2 = {"configurable": {"thread_id": "session-rome-002"}}
        query2 = "Перевір погоду в Празі у грудні та переведи 250 EUR в UAH."
        final_session_2 = app_after_restart.invoke(
            create_initial_state(query2),
            config=config_session_2,
        )
        _print_result_summary(final_session_2)

        # Перевірка ізоляції станів між двома thread_id
        st1 = app_after_restart.get_state(config_session_1).values
        st2 = app_after_restart.get_state(config_session_2).values
        print("\n[Перевірка незалежності thread_id у agent_state.db]")
        print(f"  session-paris-001 -> goal='{st1['goal']}', steps={st1['current_step']}, completed={st1['completed']}")
        print(f"  session-rome-002  -> goal='{st2['goal']}', steps={st2['current_step']}, completed={st2['completed']}")
        assert st1["goal"] != st2["goal"], "Стани різних thread_id мають бути незалежними!"
        print("  OK: Стани thread_id повністю ізольовані та збережені на диск.")
    finally:
        conn2.close()


# ── ЗАВДАННЯ 3: Agentic RAG з ChromaDB ───────────────────────────────────────

def demo_task3_agentic_rag() -> None:
    _banner("ЗАВДАННЯ 3: Agentic RAG із ChromaDB (автономний вибір RAG vs інші tools)")

    collection = get_knowledge_collection()
    print(f"База знань ChromaDB ініціалізована: {collection.count()} документів.")

    # Прямий тест інструмента search_knowledge
    print("\n[Прямий тест `search_knowledge`]")
    direct_rag = search_knowledge.invoke({"query": "правила безвізового режиму ЄС для українців", "top_k": 3})
    print(direct_rag)

    # Перевірка автономного вибору інструментів агентом
    llm = get_llm()
    tools = [*SAFE_TOOLS, search_knowledge]
    app = build_plan_execute_graph(tools=tools, llm=llm)

    rag_tests = [
        (
            "Тест A (Довідковий запит -> очікується RAG `search_knowledge`)",
            "Які правила безвізу до ЄС для громадян України, скільки коштує квиток у Лувр і в який день він закритий?",
        ),
        (
            "Тест B (Обчислення + валюта -> очікуються `calculate`/`convert_currency`, БЕЗ RAG)",
            "Порахуй точну суму: (125 * 4) + 45 + 30 в EUR і переведи результат у гривні (UAH).",
        ),
        (
            "Тест C (Гібридний запит -> очікується поєднання RAG + action tools)",
            "Які габарити безкоштовної ручної поклажі у Ryanair? Також перевір погоду в Барселоні у жовтні "
            "та порахуй вартість 3 ночей по 125 EUR.",
        ),
    ]

    for label, query in rag_tests:
        print(f"\n--- {label} ---")
        print(f"Запит: {query}")
        res = app.invoke(create_initial_state(query))
        print(f"Агент обрав інструменти: {res.get('tools_used', [])}")
        _print_result_summary(res)


# ── ЗАВДАННЯ 4: Human-in-the-Loop (interrupt / approve / reject / edit) ──────

def demo_task4_hitl() -> None:
    _banner("ЗАВДАННЯ 4: Human-in-the-Loop (interrupt / approve / reject / edit)")

    BOOKING_LEDGER.clear()
    llm = get_llm()

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        app_hitl = build_hitl_graph(checkpointer=saver, llm=llm)

        # ── Сценарій 1: APPROVE ──────────────────────────────────────────────
        print("\n--- Сценарій 1: APPROVE (підтвердження бронювання оператором) ---")
        cfg_approve = {"configurable": {"thread_id": "hitl-approve-001"}}
        query_approve = (
            "Знайди у Парижі готель 'Hotel Ibis Paris Bastille' (115 EUR/ніч), "
            "порахуй вартість 3 ночей і забронюй його з датою заїзду 2026-05-15."
        )
        paused = app_hitl.invoke(create_initial_state(query_approve), config=cfg_approve)
        interrupts = paused.get("__interrupt__")
        if interrupts:
            print("Граф зупинено на interrupt()! Деталі запиту на підтвердження:")
            print(interrupts[0].value["message"])
            print(f"Кількість записів у BOOKING_LEDGER ДО підтвердження: {len(BOOKING_LEDGER)}")

            print("\n>>> Рішення людини: APPROVE")
            resumed = app_hitl.invoke(Command(resume={"action": "approve"}), config=cfg_approve)
            print(f"Кількість записів у BOOKING_LEDGER ПІСЛЯ підтвердження: {len(BOOKING_LEDGER)}")
            _print_result_summary(resumed)
        else:
            print("УВАГА: interrupt() не спрацював у сценарії approve.")

        # ── Сценарій 2: REJECT ───────────────────────────────────────────────
        print("\n--- Сценарій 2: REJECT (відхилення бронювання оператором) ---")
        ledger_before_reject = len(BOOKING_LEDGER)
        cfg_reject = {"configurable": {"thread_id": "hitl-reject-002"}}
        query_reject = (
            "Забронюй готель 'Le Marais Boutique Hotel' у місті Париж на 4 ночі "
            "з заїздом 2026-06-10 загальною вартістю 880 EUR."
        )
        paused_rej = app_hitl.invoke(create_initial_state(query_reject), config=cfg_reject)
        interrupts_rej = paused_rej.get("__interrupt__")
        if interrupts_rej:
            print("Граф зупинено на interrupt()! Деталі запиту на підтвердження:")
            print(interrupts_rej[0].value["message"])

            print("\n>>> Рішення людини: REJECT (Причина: Занадто дорого для нашого бюджету)")
            resumed_rej = app_hitl.invoke(
                Command(resume={"action": "reject", "reason": "Занадто дорого для нашого бюджету"}),
                config=cfg_reject,
            )
            print(
                f"Кількість записів у BOOKING_LEDGER: до={ledger_before_reject}, "
                f"після={len(BOOKING_LEDGER)} (бронювання НЕ виконано!)"
            )
            _print_result_summary(resumed_rej)
        else:
            print("УВАГА: interrupt() не спрацював у сценарії reject.")

        # ── Сценарій 3: EDIT ─────────────────────────────────────────────────
        print("\n--- Сценарій 3: EDIT (редагування параметрів бронювання оператором) ---")
        ledger_before_edit = len(BOOKING_LEDGER)
        cfg_edit = {"configurable": {"thread_id": "hitl-edit-003"}}
        query_edit = (
            "Забронюй готель 'Hotel Artemide Express' у місті Рим на 5 ночей "
            "з заїздом 2026-09-01 загальною вартістю 550 EUR."
        )
        paused_edit = app_hitl.invoke(create_initial_state(query_edit), config=cfg_edit)
        interrupts_edit = paused_edit.get("__interrupt__")
        if interrupts_edit:
            print("Граф зупинено на interrupt()! Початкові параметри від агента:")
            print(interrupts_edit[0].value["message"])

            edited_params = {"nights": 3, "total_cost_eur": 330.0}
            print(f"\n>>> Рішення людини: EDIT -> {edited_params}")
            resumed_edit = app_hitl.invoke(
                Command(resume={"action": "edit", "args": edited_params}),
                config=cfg_edit,
            )
            print(
                f"Кількість записів у BOOKING_LEDGER: до={ledger_before_edit}, "
                f"після={len(BOOKING_LEDGER)}"
            )
            if BOOKING_LEDGER:
                print(f"Останній запис у реєстрі бронювань: {BOOKING_LEDGER[-1]}")
            _print_result_summary(resumed_edit)
        else:
            print("УВАГА: interrupt() не спрацював у сценарії edit.")

        print("\nПідсумковий реєстр фактичних бронювань (BOOKING_LEDGER):")
        for rec in BOOKING_LEDGER:
            print(f"  - {rec}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ДЗ №2: Plan-and-Execute агент у LangGraph з Memory, Agentic RAG та HITL"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Яке завдання запустити (1, 2, 3, 4 або all).",
    )
    args = parser.parse_args()

    print(f"Запуск ДЗ №2 | LLM модель: {GEMINI_MODEL} | SQLite DB: {DB_PATH.name}")

    if args.task in ("1", "all"):
        demo_task1_plan_execute()
    if args.task in ("2", "all"):
        demo_task2_persistence(reset_db=(args.task in ("2", "all")))
    if args.task in ("3", "all"):
        demo_task3_agentic_rag()
    if args.task in ("4", "all"):
        demo_task4_hitl()


if __name__ == "__main__":
    main()
