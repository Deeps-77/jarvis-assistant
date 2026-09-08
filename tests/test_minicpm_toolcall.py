"""Benchmark MiniCPM tool-calling vs direct-answer scenarios.

Calls the real Ollama model — no mocks. Measures timing, tool calls made,
and output quality for 10 scenarios across 4 categories.

Usage:  python tests/test_minicpm_toolcall.py
         python tests/test_minicpm_toolcall.py --model some-other-model
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

try:
    from chat_tools import (
        web_search,
        get_current_time,
        date_calculator,
        get_weather,
        get_exchange_rate,
        get_crypto_price,
    )
except ImportError as e:
    print(f"WARNING: could not import chat_tools: {e}")
    web_search = None
    get_current_time = None
    date_calculator = None
    get_weather = None
    get_exchange_rate = None
    get_crypto_price = None

ALL_TOOLS = [
    t for t in [
        web_search,
        get_current_time,
        date_calculator,
        get_weather,
        get_exchange_rate,
        get_crypto_price,
    ] if t is not None
]

SYSTEM_PROMPT = """You are Jarvis, a personal AI assistant.
The current date and time is provided in a system message at the start of every turn - trust it above all other sources.
NEVER call web_search for the current date, day of the week, or clock time; answer those directly from the provided date.
For math, logic puzzles, coding, translation, definitions of common concepts, and creative writing: answer directly from your own knowledge and NEVER call web_search.
Use web_search ONLY when the question needs real-time information: news, prices, weather, sports scores, product rankings, or recent events.
Dedicated tools give exact live facts and are preferred over web_search when they match: get_current_time for time or date anywhere in the world, date_calculator for calendar math, get_weather for current weather, get_exchange_rate for currency rates, get_crypto_price for cryptocurrency prices.
When you do search: call web_search at most twice per question, never repeat a query you already tried, then answer using ONLY the results and cite them like [1][2].
If search results don't contain the answer, say you don't know instead of guessing.
Never invent facts, dates, or numbers.
Remember: you are Jarvis."""

SCENARIOS = [
    ("A-no-tool", "Recommend some good web series which is fast paced and with good story", None, "Direct answer expected (original failing query)"),
    ("A-no-tool", "What is the capital of France?", None, "Simple factual — no tool needed"),
    ("A-no-tool", "Translate hello to Spanish", None, "Translation — no tool needed"),
    ("A-no-tool", "Write a short poem about rain", None, "Creative writing — no tool needed"),
    ("B-single-tool", "What time is it in Tokyo right now?", "get_current_time", "Should use get_current_time"),
    ("B-single-tool", "What's the current weather in Mumbai?", "get_weather", "Should use get_weather"),
    ("B-single-tool", "What is 1 Bitcoin in USD right now?", "get_crypto_price", "Should use get_crypto_price"),
    ("C-web-search", "What is the latest news about artificial intelligence?", "web_search", "Should use web_search for recent news"),
    ("C-web-search", "Who won the last Formula 1 Grand Prix?", "web_search", "Should use web_search for sports"),
    ("D-multi", "What's the weather in Paris and what is the USD to EUR exchange rate?", "get_weather+get_exchange_rate", "Should call both tools"),
]


def _check_ollama(model: str) -> bool:
    try:
        url = "http://localhost:11434/api/tags"
        resp = urllib.request.urlopen(url, timeout=5)
        data = json.loads(resp.read())
        names = [m["name"] for m in data.get("models", [])]
        return any(model in n for n in names)
    except Exception:
        return False


import asyncio


async def _run_with_tools(llm, prompt: str, max_rounds: int = 4) -> dict:
    agent = create_react_agent(llm, tools=ALL_TOOLS)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    config = {"recursion_limit": max_rounds * 2 + 4}

    t0 = time.perf_counter()
    all_tool_calls = []
    final_content = ""
    rounds = 0

    try:
        async for update in agent.astream({"messages": messages}, config=config):
            for node_output in update.values():
                if isinstance(node_output, dict) and "messages" in node_output:
                    for msg in node_output["messages"]:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                all_tool_calls.append(tc["name"])
                            rounds += 1
                        if hasattr(msg, "content") and msg.content:
                            final_content = msg.content
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "elapsed": elapsed,
            "tool_calls": all_tool_calls,
            "rounds": rounds,
            "output": f"ERROR: {e}",
            "output_len": 0,
        }

    elapsed = time.perf_counter() - t0
    return {
        "elapsed": elapsed,
        "tool_calls": all_tool_calls,
        "rounds": rounds,
        "output": final_content[:500],
        "output_len": len(final_content),
    }


def _run_without_tools(llm, prompt: str) -> dict:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    t0 = time.perf_counter()
    try:
        resp = llm.invoke(messages)
        elapsed = time.perf_counter() - t0
        content = getattr(resp, "content", "") or ""
        return {
            "elapsed": elapsed,
            "output": content[:500],
            "output_len": len(content),
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "elapsed": elapsed,
            "output": f"ERROR: {e}",
            "output_len": 0,
        }


def _verdict(cat: str, expected: str | None, tool_calls: list[str]) -> str:
    if expected is None:
        if not tool_calls:
            return "PASS"
        return f"FAIL (called {tool_calls} but should use NO tools)"
    if "+" in expected:
        needed = set(expected.split("+"))
        got = set(tool_calls)
        if needed.issubset(got):
            return "PASS"
        missing = needed - got
        return f"FAIL (missing {missing})"
    if expected in tool_calls:
        return "PASS"
    if not tool_calls:
        return f"FAIL (expected {expected}, used NO tools)"
    return f"FAIL (expected {expected}, got {tool_calls})"


def main():
    parser = argparse.ArgumentParser(description="Benchmark MiniCPM tool-calling")
    parser.add_argument("--model", default="hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M")
    parser.add_argument("--timeout", type=int, default=300, help="Per-scenario timeout in seconds")
    args = parser.parse_args()

    model = args.model
    print(f"Model: {model}")
    print(f"Ollama check: {'OK' if _check_ollama(model) else 'NOT FOUND'}")
    print(f"Tools loaded: {len(ALL_TOOLS)}")
    print("=" * 80)

    llm = ChatOllama(model=model, temperature=0.2, num_ctx=8192, timeout=600, keep_alive=-1)

    results = []
    for i, (cat, prompt, expected, desc) in enumerate(SCENARIOS, 1):
        print(f"\n[{i:2d}/{len(SCENARIOS)}] ({cat}) {desc}")
        print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

        print(f"  Running with tools...", end=" ", flush=True)
        agent_result = asyncio.run(_run_with_tools(llm, prompt))
        verdict = _verdict(cat, expected, agent_result["tool_calls"])
        print(f"{agent_result['elapsed']:.1f}s | tools={agent_result['tool_calls']} | {verdict}")

        print(f"  Running without tools...", end=" ", flush=True)
        notool_result = _run_without_tools(llm, prompt)
        print(f"{notool_result['elapsed']:.1f}s | {notool_result['output_len']} chars")

        results.append({
            "category": cat,
            "prompt": prompt,
            "expected": expected,
            "description": desc,
            "with_tools": agent_result,
            "without_tools": notool_result,
            "verdict": verdict,
        })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    pass_count = sum(1 for r in results if r["verdict"] == "PASS")
    fail_count = len(results) - pass_count
    avg_agent_time = sum(r["with_tools"]["elapsed"] for r in results) / len(results)
    avg_notool_time = sum(r["without_tools"]["elapsed"] for r in results) / len(results)

    for cat in ["A-no-tool", "B-single-tool", "C-web-search", "D-multi"]:
        cat_results = [r for r in results if r["category"] == cat]
        cat_pass = sum(1 for r in cat_results if r["verdict"] == "PASS")
        print(f"  {cat:15s}: {cat_pass}/{len(cat_results)} passed")

    print(f"\n  Total: {pass_count}/{len(results)} passed, {fail_count} failed")
    print(f"  Avg agent time: {avg_agent_time:.1f}s")
    print(f"  Avg no-tool time: {avg_notool_time:.1f}s")
    print(f"  Tool-call overhead: {avg_agent_time - avg_notool_time:.1f}s avg")

    if fail_count:
        print(f"\n  FAILURES:")
        for r in results:
            if r["verdict"] != "PASS":
                print(f"    [{r['category']}] {r['prompt'][:60]}")
                print(f"      {r['verdict']}")
                print(f"      Agent output: {r['with_tools']['output'][:200]}")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
