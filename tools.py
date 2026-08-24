import logging
from datetime import datetime, timedelta
from typing import Annotated
from zoneinfo import ZoneInfo

import httpx
from ddgs import DDGS
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

logger = logging.getLogger(__name__)

MAX_SEARCH_RESULTS = 5

_DOC_STORE = None


def set_doc_store(store) -> None:
    global _DOC_STORE
    _DOC_STORE = store


def _cfg_owner(config) -> str | None:
    try:
        return ((config or {}).get("configurable") or {}).get("doc_owner")
    except Exception:
        return None


@tool
def web_search(query: str) -> str:
    """Searches the web for current information. Use for news, prices, weather,
    sports scores, product comparisons, or any fact you are unsure about."""
    try:
        results = list(DDGS(timeout=20).text(query, max_results=MAX_SEARCH_RESULTS))
    except Exception:
        return "ERROR: web search is unavailable right now."
    if not results:
        return "No useful results found for this query."
    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(f"[{i}] {r.get('title', '')}\n{r.get('body', '')}\nURL: {r.get('href', '')}")
    return "\n\n".join(parts)


@tool
def get_current_time(timezone: str = "") -> str:
    """Returns the exact current date and time. Optionally pass an IANA timezone
    name like 'Asia/Kolkata', 'America/New_York', or 'Europe/London' for a specific
    zone. Use for ANY question about the current time or date in any location."""
    try:
        if timezone.strip():
            now = datetime.now(ZoneInfo(timezone.strip()))
            label = timezone.strip()
        else:
            now = datetime.now().astimezone()
            label = now.tzname() or "local"
        return f"{now:%A, %d %B %Y}, {now:%I:%M:%S %p} ({label}, UTC{now:%z})"
    except Exception:
        return (
            f"ERROR: unknown timezone '{timezone}'. "
            "Use IANA names like 'Asia/Kolkata' or 'America/New_York'."
        )


@tool
def date_calculator(operation: str, date1: str, date2: str = "", days: int = 0) -> str:
    """Calendar math. operation must be one of: 'days_between' (days from date1
    to date2), 'add_days' (date1 plus days; negative counts backwards), or
    'weekday_of' (weekday name of date1). Dates use YYYY-MM-DD format."""
    try:
        d1 = datetime.strptime(date1.strip(), "%Y-%m-%d").date()
    except ValueError:
        return f"ERROR: could not parse date1='{date1}'. Use YYYY-MM-DD format."
    op = operation.strip().lower()
    try:
        if op == "weekday_of":
            return f"{date1} is a {d1:%A}."
        if op == "add_days":
            d3 = d1 + timedelta(days=int(days))
            return f"{date1} plus {days} days = {d3.isoformat()} ({d3:%A})."
        if op == "days_between":
            d2 = datetime.strptime(date2.strip(), "%Y-%m-%d").date()
            n = (d2 - d1).days
            if n == 0:
                return "The two dates are the same day."
            direction = "after" if n > 0 else "before"
            return f"{d2:%A, %d %B %Y} is {abs(n)} days {direction} {d1:%A, %d %B %Y}."
    except ValueError:
        return f"ERROR: could not parse date2='{date2}'. Use YYYY-MM-DD format."
    return f"ERROR: unknown operation '{operation}'. Use days_between, add_days, or weekday_of."


@tool
def get_weather(location: str) -> str:
    """Returns CURRENT live weather conditions and today's range for any city
    worldwide. Use this for ANY question about temperature, rain, humidity,
    wind, or weather instead of web_search."""
    try:
        r = httpx.get(
            f"https://wttr.in/{location.strip().replace(' ', '+')}?format=j1",
            timeout=10,
            follow_redirects=True,
        )
        r.raise_for_status()
        data = r.json()
        cc = data["current_condition"][0]
        area = data.get("nearest_area", [{}])[0]
        name = area.get("areaName", [{"value": location}])[0]["value"]
        country = area.get("country", [{"value": ""}])[0]["value"]
        desc = cc["weatherDesc"][0]["value"]
        today = data["weather"][0]
        lines = [
            f"Weather in {name}, {country} (observed {cc['observation_time']} UTC):",
            f"- Condition: {desc}",
            f"- Temperature: {cc['temp_C']}C (feels like {cc['FeelsLikeC']}C)",
            f"- Humidity: {cc['humidity']}%",
            f"- Wind: {cc['winddir16Point']} at {cc['windspeedKmph']} km/h",
            f"- Precipitation today: {cc['precipMM']} mm",
            f"- Today's range: {today['mintempC']}C to {today['maxtempC']}C",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: could not fetch live weather for '{location}' ({e})."


@tool
def get_exchange_rate(base_currency: str, target_currency: str) -> str:
    """Returns the current exchange rate between two currencies. Use ISO codes
    like USD, EUR, INR, GBP, JPY."""
    base = base_currency.strip().upper()
    target = target_currency.strip().upper()
    try:
        r = httpx.get(
            f"https://api.frankfurter.dev/v1/latest?base={base}&symbols={target}",
            timeout=10,
        )
        r.raise_for_status()
        payload = r.json()
        rate = payload["rates"][target]
        return f"1 {base} = {rate} {target} (ECB reference rate for {payload['date']})"
    except Exception:
        return f"ERROR: could not fetch rate {base}->{target}. Check ISO currency codes."


@tool
def get_crypto_price(coin_id: str) -> str:
    """Returns the current price of a cryptocurrency in USD and INR. Use
    CoinGecko ids like 'bitcoin', 'ethereum', 'solana', or 'dogecoin'."""
    coin = coin_id.strip().lower()
    try:
        r = httpx.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": coin, "vs_currencies": "usd,inr"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if coin not in data:
            known = ", ".join(list(data.keys())[:5]) or "none matched"
            return f"ERROR: unknown coin id '{coin_id}'. Try ids like bitcoin, ethereum, solana. ({known})"
        usd = data[coin].get("usd")
        inr = data[coin].get("inr")
        return f"{coin}: ${usd:,.2f} USD / INR {inr:,.0f}"
    except Exception as e:
        return f"ERROR: could not fetch crypto price ({e})."


@tool
async def search_documents(query: str, config: Annotated[RunnableConfig, InjectedToolArg]) -> str:
    """Searches inside the documents the user has uploaded (PDF, DOCX, TXT, MD).
    Use for ANY question about the contents of the user's own files instead of web_search."""
    if _DOC_STORE is None or not _DOC_STORE.enabled:
        return "ERROR: document search is unavailable."
    owner = _cfg_owner(config)
    if not owner:
        return "ERROR: no document owner context."
    hits = await _DOC_STORE.search(owner, query)
    if not hits:
        return "No matching passages found in the uploaded documents."
    blocks = [f"[{src} · chunk {idx}]\n{text}" for src, idx, text in hits]
    return "\n\n".join(blocks)


@tool
async def list_documents(config: Annotated[RunnableConfig, InjectedToolArg]) -> str:
    """Lists every document the user has uploaded, with chunk counts.
    Use when asked what files or documents exist."""
    if _DOC_STORE is None or not _DOC_STORE.enabled:
        return "ERROR: document storage is unavailable."
    owner = _cfg_owner(config)
    if not owner:
        return "ERROR: no document owner context."
    docs = await _DOC_STORE.list_docs(owner)
    if not docs:
        return "No documents uploaded yet."
    return "\n".join(f"- {d['source']} ({d['chunks']} chunks, {d['date']})" for d in docs)


@tool
async def summarize_document(source: str, config: Annotated[RunnableConfig, InjectedToolArg]) -> str:
    """Returns the readable text of ONE uploaded document by its exact name from
    list_documents. Use for summarize/extract requests about a specific file."""
    if _DOC_STORE is None or not _DOC_STORE.enabled:
        return "ERROR: document storage is unavailable."
    owner = _cfg_owner(config)
    if not owner:
        return "ERROR: no document owner context."
    text, truncated = await _DOC_STORE.doc_text(owner, source)
    if not text:
        return f"ERROR: '{source}' not found or empty. Use list_documents for exact names."
    note = "\n\n(NOTE: document was longer than this excerpt.)" if truncated else ""
    return f"TEXT OF {source}:\n\n{text}{note}"