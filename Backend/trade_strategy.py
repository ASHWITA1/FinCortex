import os
import json
import math
import warnings
import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import requests
from contextlib import redirect_stdout, redirect_stderr
import io

import matplotlib
matplotlib.use("Agg")   # use non-interactive backend — prevents Tk/Tcl GUI issues
import matplotlib.pyplot as plt


def capture_output(func, *args, **kwargs):
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            func(*args, **kwargs)
    except Exception as e:
        buffer.write(str(e))
    return buffer.getvalue()


try:
    from config_db import log_to_db
except Exception:
    log_to_db=None

warnings.filterwarnings("ignore")
logging.getLogger("autogen").setLevel(logging.ERROR)
logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)
logging.getLogger("pydantic").setLevel(logging.ERROR)

try:
    import autogen
except Exception:
    autogen = None

# ======================================================
# CONFIG / KEYS
# ======================================================
def load_api_key(path: str = "config_api_keys"):
    """
    Load OLLAMA_API_KEY from config_api_keys or config_api_keys.json
    If env var OLLAMA_API_KEY is set, that wins.
    """
    env_key = os.getenv("OLLAMA_API_KEY")
    if env_key:
        return env_key

    if not os.path.exists(path) and not os.path.exists(path + ".json"):
        raise FileNotFoundError(f"Config file '{path}' not found.")

    p = path if os.path.exists(path) else path + ".json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    key = data.get("OLLAMA_API_KEY")
    if not key:
        raise KeyError("OLLAMA_API_KEY not found in config_api_keys.")
    return key


def load_all_keys(path: str = "config_api_keys"):
    if not os.path.exists(path) and not os.path.exists(path + ".json"):
        return {}
    try:
        p = path if os.path.exists(path) else path + ".json"
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# ======================================================
# LLM CONFIG (OLLAMA CLOUD, OPENAI-COMPATIBLE)
# ======================================================
def setup_llm_config_ollama(ollama_api_key: str,
                            model: str = "gpt-oss:20b") -> dict:
    """
    autogen-compatible llm_config for Ollama Cloud.

    For cloud OpenAI-compatible endpoint:
        base_url = "https://ollama.com/v1"
        api_key  = OLLAMA_API_KEY
        api_type = "openai"

    You can change model to any cloud-supported:
        e.g. "gpt-oss:120b-cloud", "llama3.1", etc.
    """
    if not ollama_api_key:
        raise ValueError("OLLAMA_API_KEY is required.")

    return {
        "config_list": [
            {
                "model": model,
                "api_key": ollama_api_key,
                "base_url": "https://ollama.com/v1",
                "api_type": "openai",
            }
        ],
        "timeout": 120,
        "temperature": 0.2,
        "max_tokens": 1200,
    }

# ======================================================
# DATA FETCHING
# ======================================================
def get_realtime_price_finnhub(ticker: str, finnhub_key: str):
    if not finnhub_key:
        return None
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={finnhub_key}"
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            j = r.json()
            price = j.get("c")
            if price and price > 0:
                return float(price)
    except Exception:
        pass
    return None


def fetch_ohlc(ticker, days=250):
    df = yf.Ticker(ticker).history(
        period=f"{days}d", interval="1d", auto_adjust=False
    )
    if df.empty:
        raise ValueError(f"No market data available for {ticker}.")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df.astype(float)


def get_company_info(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "beta": info.get("beta", "N/A"),
            "debt_equity": info.get("debtToEquity", "N/A"),
            "current_ratio": info.get("currentRatio", "N/A"),
            "roe": info.get("returnOnEquity", "N/A"),
        }
    except Exception:
        return {
            "name": ticker,
            "sector": "N/A",
            "industry": "N/A",
            "market_cap": 0,
            "pe_ratio": "N/A",
            "beta": "N/A",
            "debt_equity": "N/A",
            "current_ratio": "N/A",
            "roe": "N/A",
        }


def calculate_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50.0


def calculate_ma(df, period=50):
    return df["Close"].rolling(window=period).mean().iloc[-1]

# ======================================================
# BUILD PROMPT (KEEP SAME OUTPUT STRUCTURE)
# ======================================================
def build_prompt(ticker, amount, risk, df, company_info):
    keys = load_all_keys()
    finnhub_key = keys.get("FINNHUB_API_KEY", "").strip()

    close = float(df["Close"].iloc[-1])
    realtime = get_realtime_price_finnhub(ticker, finnhub_key)
    close_used = realtime if realtime is not None else close

    high10 = float(df["High"].tail(10).max())
    low10 = float(df["Low"].tail(10).min())
    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else close_used
    change = ((close_used - prev) / prev * 100) if prev else 0.0

    rsi = calculate_rsi(df)
    ma50 = calculate_ma(df, 50)

    volume_recent = df["Volume"].tail(10).mean()
    volume_prev = df["Volume"].tail(20).head(10).mean()
    volume_change = (
        (volume_recent - volume_prev) / volume_prev * 100 if volume_prev else 0.0
    )

    numeric_summary = f"""
Stock: {ticker} - {company_info['name']}
Sector: {company_info['sector']} | Industry: {company_info['industry']}
Current Price: ${close_used:.2f}
10-day High: ${high10:.2f}
10-day Low: ${low10:.2f}
Daily Change: {change:+.2f}%
RSI (14): {rsi:.1f}
50-day MA: ${ma50:.2f}
Price vs MA: {'Above' if close_used > ma50 else 'Below'}
Volume Trend: {volume_change:+.1f}%
Investment Amount: ${amount}
Risk Tolerance: {risk}
Market Cap: ${company_info['market_cap']:,.0f}
P/E Ratio: {company_info['pe_ratio']}
Beta: {company_info['beta']}
Debt-to-Equity: {company_info['debt_equity']}
Current Ratio: {company_info['current_ratio']}
ROE: {company_info['roe']}
"""

    # *** DO NOT CHANGE THIS STRUCTURE ***
    prompt = f"""
You are an expert trading analyst. Using the data below, create a comprehensive trading strategy.

Provide your response in EXACTLY this structure with detailed reasoning:

ENTRY STRATEGY
Buy Price: $[number]
Entry Timing: [specific guidance]
Position Size: [number] shares
Total Cost: $[number]
Entry Justification: [2-3 sentences explaining why this entry point]

EXIT STRATEGY
Target 1: $[number] (Sell [%] of position)
Target 1 Rationale: [1-2 sentences]
Target 2: $[number] (Sell [%] of position)
Target 2 Rationale: [1-2 sentences]
Expected Gain: $[number] ([%])
Exit Timing: [guidance]

RISK MANAGEMENT
Stop Loss: $[number]
Stop Loss Rationale: [1-2 sentences]
Max Loss: $[number]
Risk/Reward: [ratio]
Position Hedging: [strategy if applicable]

POSITION SIZING
Portfolio Allocation: [%]
Justification: [1-2 sentences]
Remaining Capital: [%]

TIMELINE
Entry Window: [specific dates/conditions]
Target Timeframe: [weeks]
Review Points: [key dates]

TECHNICAL ANALYSIS
RSI Signal: [interpretation]
Moving Average: [interpretation]
Volume Analysis: [interpretation]
Support/Resistance: [key levels]

CATALYSTS & RISKS
Positive Catalysts: [2-3 points]
Risk Factors: [2-3 points]
Market Conditions: [assessment]

EXECUTION PLAN
Step 1: [action]
Step 2: [action]
Step 3: [action]
Monitoring: [what to watch]

Numeric Summary:
{numeric_summary}
"""
    return prompt

# ======================================================
# AUTOGEN CALL (OLLAMA CLOUD)
# ======================================================
def call_autogen(llm_config: dict, prompt: str) -> str:
    if autogen is None:
        raise RuntimeError("autogen is required. Install pyautogen.")

    assistant = autogen.AssistantAgent(
        name="Trade_Analyst",
        llm_config=llm_config,
        system_message=(
            "You are an expert trading analyst. Provide comprehensive, professional "
            "trading strategies with detailed explanations for each recommendation. "
            "Strictly follow the requested output structure and headings."
        ),
    )
    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    print("\n🤖 Running AI trade strategy analysis (Ollama Cloud).")
    user_proxy.initiate_chat(assistant, message=prompt, clear_history=True)

    last = user_proxy.last_message()
    if not last:
        raise RuntimeError("LLM did not return a response.")
    return last.get("content", "").strip()

# ======================================================
# PARSING STRATEGY OUTPUT
# ======================================================
def extract_number(text):
    import re
    m = re.search(r"(-?\d[\d,\.]*)", text)
    if not m:
        return "0"
    return m.group(1).replace(",", "")


def parse_enhanced_strategy(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    parsed = {
        "entry": {},
        "exit": {},
        "risk": {},
        "position": {},
        "timeline": {},
        "technical": {},
        "catalysts": {},
        "execution": {},
        "notes": [],
    }

    section = None

    for line in lines:
        up = line.upper()

        if up.startswith("ENTRY STRATEGY"):
            section = "entry"
            continue
        elif up.startswith("EXIT STRATEGY"):
            section = "exit"
            continue
        elif up.startswith("RISK MANAGEMENT"):
            section = "risk"
            continue
        elif up.startswith("POSITION SIZING"):
            section = "position"
            continue
        elif up.startswith("TIMELINE"):
            section = "timeline"
            continue
        elif up.startswith("TECHNICAL ANALYSIS"):
            section = "technical"
            continue
        elif up.startswith("CATALYSTS"):
            section = "catalysts"
            continue
        elif up.startswith("EXECUTION PLAN"):
            section = "execution"
            continue

        try:
            if section == "entry":
                if "buy price" in line.lower():
                    parsed["entry"]["buy_price"] = float(extract_number(line))
                if "position size" in line.lower() or "shares" in line.lower():
                    parsed["entry"]["shares"] = int(float(extract_number(line)))
                if "total cost" in line.lower():
                    parsed["entry"]["total_cost"] = float(extract_number(line))
                if "justification" in line.lower() or "rationale" in line.lower():
                    parsed["entry"]["justification"] = line.split(":", 1)[-1].strip()

            if section == "exit":
                if "target 1" in line.lower() and "$" in line:
                    parsed["exit"]["t1"] = float(extract_number(line))
                if "target 2" in line.lower() and "$" in line:
                    parsed["exit"]["t2"] = float(extract_number(line))
                if "expected gain" in line.lower():
                    parsed["exit"]["gain"] = float(extract_number(line))

            if section == "risk":
                if "stop loss" in line.lower() and "$" in line:
                    parsed["risk"]["stop"] = float(extract_number(line))
                if "max loss" in line.lower():
                    parsed["risk"]["max_loss"] = float(extract_number(line))
                if "risk/reward" in line.lower():
                    parsed["risk"]["rr"] = line.split(":", 1)[-1].strip()

            if section == "position":
                if "allocation" in line.lower():
                    parsed["position"]["alloc"] = float(extract_number(line))
                if "remaining" in line.lower():
                    parsed["position"]["remain"] = float(extract_number(line))
        except Exception:
            # ignore parse errors; we'll fill defaults later
            pass

    return parsed


def ensure_numbers(parsed, amount, df):
    price = float(df["Close"].iloc[-1])
    amount = float(amount)

    parsed["entry"].setdefault("buy_price", price)
    if parsed["entry"].get("buy_price", 0) > 0:
        if amount < parsed["entry"]["buy_price"]:
            parsed["entry"].setdefault(
                "shares", round(amount / parsed["entry"]["buy_price"], 6)
            )
        else:
            parsed["entry"].setdefault(
                "shares", max(1, int(amount // parsed["entry"]["buy_price"]))
            )
    else:
        parsed["entry"].setdefault("shares", 0)

    parsed["entry"].setdefault(
        "total_cost",
        round(parsed["entry"]["shares"] * parsed["entry"]["buy_price"], 2),
    )
    parsed["exit"].setdefault("t1", round(price * 1.02, 2))
    parsed["exit"].setdefault("t2", round(price * 1.05, 2))

    try:
        if isinstance(parsed["entry"]["shares"], int):
            half_int = parsed["entry"]["shares"] // 2
            gain = (parsed["exit"]["t1"] - price) * half_int + (
                parsed["exit"]["t2"] - price
            ) * (parsed["entry"]["shares"] - half_int)
        else:
            s = parsed["entry"]["shares"]
            sell1 = round(s * 0.33, 6)
            sell2 = round(s - sell1, 6)
            gain = (parsed["exit"]["t1"] - price) * sell1 + (
                parsed["exit"]["t2"] - price
            ) * sell2
    except Exception:
        gain = 0.0

    parsed["exit"].setdefault("gain", round(gain, 2))
    parsed["risk"].setdefault("stop", round(price * 0.97, 2))
    parsed["risk"].setdefault(
        "max_loss",
        round((price - parsed["risk"]["stop"]) * parsed["entry"]["shares"], 2),
    )
    parsed["risk"].setdefault(
        "rr",
        f"{(parsed['exit']['gain'] / parsed['risk']['max_loss']):.2f}"
        if parsed["risk"]["max_loss"]
        else "∞",
    )
    used = parsed["entry"]["total_cost"]

    if amount > 0:
        alloc_pct = round((used / amount) * 100, 2)
        remain_pct = round(100 - alloc_pct, 2)
    else:
        alloc_pct = 0
        remain_pct = 100

    parsed["position"]["alloc"] = alloc_pct
    parsed["position"]["remain"] = remain_pct


    return parsed

# ======================================================
# CHARTS
# ======================================================
def ensure_charts_dir():
    path = os.path.join(os.getcwd(), "charts")
    os.makedirs(path, exist_ok=True)
    return path


def entry_exit_chart(df, parsed, ticker, outdir):
    entry = parsed["entry"]["buy_price"]
    t1 = parsed["exit"]["t1"]
    t2 = parsed["exit"]["t2"]
    stop = parsed["risk"]["stop"]

    subset = df.tail(60).copy()
    figpath = os.path.join(outdir, f"{ticker}_entry_exit.png")

    try:
        mpf.plot(
            subset,
            type="candle",
            style="yahoo",
            hlines=dict(
                hlines=[entry, t1, t2, stop],
                colors=["lime", "gold", "dodgerblue", "red"],
                linestyle="--",
            ),
            savefig=figpath,
        )
    except Exception:
        plt.figure(figsize=(10, 5))
        plt.plot(subset.index, subset["Close"], label="Close")
        plt.axhline(entry, color="lime", linestyle="--", label="Entry")
        plt.axhline(t1, color="gold", linestyle="--", label="Target1")
        plt.axhline(t2, color="dodgerblue", linestyle="--", label="Target2")
        plt.axhline(stop, color="red", linestyle="--", label="Stop")
        plt.legend()
        plt.savefig(figpath)
        plt.close()

    return figpath


def risk_reward_chart(parsed, ticker, outdir):
    figpath = os.path.join(outdir, f"{ticker}_risk_reward.png")
    gain = parsed["exit"]["gain"]
    loss = parsed["risk"]["max_loss"]

    plt.figure(figsize=(6, 4))
    plt.bar(["Expected Gain", "Max Loss"], [gain, loss], color=["green", "red"])
    plt.title("Risk vs Reward Analysis")
    plt.ylabel("Amount ($)")
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()

    return figpath


# ======================================================
# PROFESSIONAL REPORT (MARKET FORECASTER STYLE)
# ======================================================
def print_professional_report(
    ticker, amount, risk, parsed, charts, df, company_info, llm_response
):
    now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    date_str = now.strftime("%d %b %Y")
    time_str = now.strftime("%H:%M IST")

    # Additional metrics
    rsi = calculate_rsi(df)
    ma50 = calculate_ma(df, 50)
    current_price = float(df["Close"].iloc[-1])

    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
    ma_status = "Bullish" if current_price > ma50 else "Bearish"

    print("\n" + "=" * 70)
    print("📈 TRADE STRATEGY REPORT ")
    # print("-" * 70)

    # print("━" * 70)
    # print(f"📅 Date: {date_str} | ⏰ Time: {time_str}")
    # print(f"📊 Stock: {ticker} - {company_info['name']}")
    # print(f"🏢 Sector: {company_info['sector']} | Industry: {company_info['industry']}")
    # print("✅ Analysis Model: Ollama Cloud (OpenAI-compatible)")
    # print("━" * 70)

    # STRATEGY OVERVIEW
    print("\n📌 Strategy Overview:")
    print("-" * 70)

    print("━" * 70)
    print(f"   Investment Amount: ${amount:,.2f}")
    print(f"   Risk Tolerance: {risk.capitalize()}")
    print(
        f"   Position Type: {'Long' if parsed['exit']['gain'] > 0 else 'Short'}"
    )

    gain_pct = (
        parsed["exit"]["gain"] / parsed["entry"]["total_cost"] * 100
        if parsed["entry"]["total_cost"]
        else 0
    )
    print(f"   Expected Return: {gain_pct:+.2f}%")
    print(f"   Confidence Level: {65 if gain_pct > 0 else 45}%")
    print()
    print("   💬 Strategy Type:")
    if gain_pct > 0:
        print("      This is a BULLISH strategy targeting price appreciation.")
        print("      Entry and exit levels are designed to capture upside momentum")
        print("      while maintaining strict risk controls.")
    else:
        print("      This is a DEFENSIVE strategy prioritizing capital preservation.")
        print("      Position sizing is conservative given current market conditions.")

    # CURRENT MARKET STATUS
    print("\n📌 Current Market Status:")
    # print("-" * 70)

    # print("━" * 70)
    print(f"   Current Price: ${current_price:.2f}")
    prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else current_price
    daily_change = (
        (current_price - prev_close) / prev_close * 100 if prev_close else 0
    )
    print(f"   Today's Change: {daily_change:+.2f}%")
    print(
        f"   Day Range: ${df['Low'].iloc[-1]:.2f} - ${df['High'].iloc[-1]:.2f}"
    )
    print(f"   Previous Close: ${prev_close:.2f}")
    print()
    print("   💬 Price Action:")
    intraday_vol = df["High"].iloc[-1] - df["Low"].iloc[-1]
    print(f"      • Intraday volatility: ${intraday_vol:.2f}")
    print(
        f"      • Opening gap: {((df['Open'].iloc[-1] - prev_close) / prev_close * 100):+.2f}%"
    )

    # ENTRY STRATEGY
    print("\n📌 Entry Strategy:")
    print("-" * 70)

    print("━" * 70)
    e = parsed["entry"]
    print(f"   Entry Price: ${e['buy_price']:.2f}")
    print(f"   Position Size: {e['shares']} shares")
    print(f"   Total Investment: ${e['total_cost']:.2f}")
    print()
    print("   💬 Entry Rationale:")
    print("      The recommended entry price represents a strategic level based on")
    print("      recent price action, technical indicators, and risk parameters.")
    print("      This price point offers favorable risk/reward while maintaining")
    print("      realistic execution probability in current market conditions.")
    print()
    print("   💬 Execution Guidance:")
    print("      • Use LIMIT orders to ensure price discipline")
    print("      • Consider scaling into position (30-30-40% splits)")
    print("      • Monitor for gaps or sudden volatility before entry")
    print("      • Set alerts at entry price for execution timing")

    # EXIT STRATEGY
    print("\n📌 Exit Strategy:")
    print("-" * 70)

    print("━" * 70)
    ex = parsed["exit"]
    if isinstance(e["shares"], int):
        half = e["shares"] // 2
    else:
        half = round(e["shares"] * 0.33, 6)
    rem_shares = e["shares"] - half

    print(f"   Target 1: ${ex['t1']:.2f} (Sell {half} shares)")
    target1_gain_pct = (ex["t1"] - e["buy_price"]) / e["buy_price"] * 100
    print(f"   Target 1 Gain: {target1_gain_pct:+.2f}%")
    print()
    print(f"   Target 2: ${ex['t2']:.2f} (Sell {rem_shares} shares)")
    target2_gain_pct = (ex["t2"] - e["buy_price"]) / e["buy_price"] * 100
    print(f"   Target 2 Gain: {target2_gain_pct:+.2f}%")
    print()
    print(f"   Total Expected Gain: ${ex['gain']:.2f} ({gain_pct:+.2f}%)")
    print()
    print("   💬 Exit Strategy Rationale:")
    print("      Targets are set using a tiered approach to balance profit-taking")
    print("      with trend participation. Target 1 secures partial profits and")
    print("      reduces position risk, while Target 2 captures extended moves.")
    print("      This structure optimizes for both certainty and opportunity.")
    print()
    print("   💬 Exit Execution:")
    print("      • Place LIMIT SELL orders at both target levels immediately")
    print("      • Consider trailing stop on remaining position after Target 1")
    print("      • Exit entirely if momentum clearly reverses before targets")
    print("      • Review targets daily against technical levels")

    # RISK MANAGEMENT
    # print("\n📌 Risk Management:")
    # print("-" * 70)

    # print("━" * 70)
    # r = parsed["risk"]
    # print(f"   Stop Loss: ${r['stop']:.2f}")
    # stop_pct = (r["stop"] - e["buy_price"]) / e["buy_price"] * 100
    # print(f"   Stop Distance: {stop_pct:.2f}%")
    # print(f"   Maximum Loss: ${r['max_loss']:.2f}")
    # print(f"   Risk/Reward Ratio: {r['rr']}")
    # print()
    # print("   💬 Risk Management Framework:")
    # print("      The stop-loss is positioned to provide protection while allowing")
    # print("      normal price fluctuation. Max loss represents total capital at risk.")
    # print("      Risk/reward ratio indicates this trade offers favorable compensation")
    # print("      for the assumed risk, meeting professional trading standards.")
    # print()
    # print("   💬 Risk Controls:")
    # print("      • NEVER move stop-loss lower after entry")
    # print("      • Use mental stops if concerned about stop-hunting")
    # print("      • Consider volatility when sizing position")
    # print("      • Exit immediately if fundamental thesis changes")
    # print(
    #     f"      • Max portfolio risk: {(r['max_loss'] / amount * 100):.1f}% of total capital"
    # )

    # POSITION SIZING
    print("\n📌 Position Sizing Analysis:")
    print("-" * 70)

    print("━" * 70)
    p = parsed["position"]
    print(f"   Portfolio Allocation: {p['alloc']:.1f}%")
    print(f"   Remaining Capital: {p['remain']:.1f}%")
    print(f"   Position Cost: ${e['total_cost']:.2f}")
    print()
    print("   💬 Sizing Rationale:")
    print("      Position size is calibrated to limit single-trade exposure while")
    print("      maintaining meaningful profit potential. This allocation preserves")
    print("      capital for other opportunities and prevents over-concentration.")
    print("      Remaining capital provides cushion for position management.")
    print()
    print("   💬 Portfolio Context:")
    print("      • Maintains diversification across positions")
    print("      • Allows for scale-in opportunities if thesis strengthens")
    print("      • Preserves dry powder for market dislocations")
    print("      • Consider correlation with existing holdings")

    # TECHNICAL ANALYSIS
    print("\n📌 Technical Analysis:")
    print("-" * 70)

    print("━" * 70)
    print(f"   RSI (14-day): {rsi:.1f} ({rsi_status})")
    print(f"   50-Day Moving Avg: ${ma50:.2f}")
    print(f"   Price vs MA50: {ma_status}")

    volume_recent = df["Volume"].tail(5).mean()
    volume_avg = df["Volume"].tail(50).mean()
    volume_trend = "Increasing" if volume_recent > volume_avg else "Decreasing"
    volume_pct = (
        (volume_recent - volume_avg) / volume_avg * 100 if volume_avg else 0
    )
    print(f"   Volume Trend: {volume_trend} ({volume_pct:+.1f}%)")
    print()
    print("   💬 Technical Interpretation:")
    print(f"      • RSI at {rsi:.0f} suggests {rsi_status.lower()} conditions")
    if rsi > 70:
        print("        (potential for pullback or consolidation)")
    elif rsi < 30:
        print("        (potential for bounce or reversal)")
    else:
        print("        (balanced momentum, room for directional move)")

    print(
        f"      • Price {'above' if current_price > ma50 else 'below'} 50-day MA indicates {ma_status.lower()} trend"
    )
    print(
        f"      • Volume {volume_trend.lower()} {abs(volume_pct):.0f}% vs average"
    )
    print(
        f"      • Support near ${df['Low'].tail(20).min():.2f} | Resistance near ${df['High'].tail(20).max():.2f}"
    )

    # FUNDAMENTAL HEALTH
    print("\n📌 Fundamental Health:")
    # print("-" * 70)

    # print("━" * 70)
    print(f"   Market Cap: ${company_info['market_cap']:,.0f}")
    print(f"   P/E Ratio: {company_info['pe_ratio']}")
    print(f"   Beta: {company_info['beta']}")
    print(f"   Debt-to-Equity: {company_info['debt_equity']}")
    print(f"   Current Ratio: {company_info['current_ratio']}")
    print(f"   ROE: {company_info['roe']}")
    print()
    print("   💬 Fundamental Assessment:")
    try:
        pe = float(company_info["pe_ratio"])
        if pe < 15:
            print("      • P/E suggests potentially undervalued relative to sector")
        elif pe > 30:
            print(
                "      • P/E indicates premium valuation, growth expectations priced in"
            )
        else:
            print("      • P/E appears reasonable for current market conditions")
    except Exception:
        print("      • P/E data unavailable for comparison")

    try:
        beta = float(company_info["beta"])
        if beta > 1.2:
            print(
                f"      • Beta of {beta:.2f} indicates higher volatility than market"
            )
        elif beta < 0.8:
            print(
                f"      • Beta of {beta:.2f} suggests lower volatility, defensive characteristics"
            )
        else:
            print(
                f"      • Beta of {beta:.2f} indicates market-average volatility"
            )
    except Exception:
        pass

    print(
        "      • Review latest earnings and guidance for fundamental changes"
    )

    # RECOMMENDED ACTION PLAN
    print("\n📌 Recommended Action Plan:")
    # print("-" * 70)

    # print("━" * 70)

    if gain_pct > 3 and rsi < 70:
        print("  1. ✅ EXECUTE - Strong buy signal with favorable risk/reward")
        print("  2. 📊 Set entry order at recommended price")
        print("  3. 🛡 Set stop-loss immediately after fill")
        print("  4. 🎯 Place target orders at both levels")
    elif gain_pct > 0:
        print("  1. ⚠️  CAUTIOUS BUY - Moderate opportunity, monitor entry carefully")
        print("  2. 📊 Wait for pullback or confirmation before entry")
        print("  3. 🛡 Use tighter stop given lower conviction")
        print("  4. 🎯 Consider taking profits earlier than targets")
    else:
        print("  1. ⏸️  WAIT - Current setup does not offer compelling opportunity")
        print("  2. 📊 Set alerts at key technical levels")
        print("  3. 🔍 Monitor for improved risk/reward setup")
        print("  4. 💼 Focus capital on higher-conviction ideas")

    # CATALYSTS & RISKS
    print("\n📌 Key Catalysts & Risk Factors:")
    print("-" * 70)

    print("━" * 70)
    print("   Positive Catalysts:")
    if ma_status == "Bullish":
        print("  ✅ Strong technical trend (price above 50-day MA)")
    if rsi < 50:
        print("  ✅ RSI suggests room for upside before overbought")
    if volume_trend == "Increasing":
        print("  ✅ Volume supporting current price action")
    print(f"  ✅ {company_info['sector']} sector positioning")

    print("\n   Risk Factors:")
    if rsi > 70:
        print("  ⚠️  Overbought RSI signals potential pullback risk")
    if ma_status == "Bearish":
        print("  ⚠️  Price below key moving average, weak trend")
    if volume_trend == "Decreasing":
        print("  ⚠️  Declining volume may signal waning momentum")
    print("  ⚠️  General market volatility and macro conditions")
    try:
        de = float(company_info["debt_equity"])
        if de > 100:
            print(f"  ⚠️  Elevated debt levels (D/E: {de:.1f})")
    except Exception:
        pass

    # TIMELINE
    weeks = (
        3 if risk.lower() == "medium"
        else 2 if risk.lower() == "low"
        else 4
    )
    review = now + timedelta(weeks=weeks)
    review_str = review.strftime("%d %b %Y")

    print("\n📌 Timeline & Monitoring:")
    print("-" * 70)

    print("━" * 70)
    print("   Entry Window: Next 1-2 trading days")
    print(f"   Target Timeframe: {weeks}-{weeks+1} weeks")
    print(f"   First Review Date: {review_str}")
    print("   Position Monitoring: Daily")
    print()
    print("   💬 Timeline Guidelines:")
    print("      The timeframe balances your risk tolerance with realistic price")
    print("      discovery patterns. Review positions at scheduled intervals to")
    print("      ensure thesis remains intact. Adjust or exit if conditions change.")
    print(
        "      Avoid premature exits based on noise, but don't ignore deterioration."
    )

    # EXECUTION CHECKLIST
    print("\n📌 Pre-Trade Execution Checklist:")
    print("-" * 70)

    r = parsed["risk"]
    print("━" * 70)
    print(f"  ☐ Verify sufficient capital for position (${e['total_cost']:.2f})")
    print("  ☐ Confirm broker liquidity for this symbol")
    print(f"  ☐ Set entry limit order at ${e['buy_price']:.2f}")
    print(f"  ☐ Prepare stop-loss order at ${r['stop']:.2f}")
    print("  ☐ Prepare target limit sells")
    print("  ☐ Document trade thesis and plan")
    print("  ☐ Set calendar reminder for review date")
    print("  ☐ Check correlation with existing positions")

    # CHART SUMMARY
    print("\n📊 Chart Analysis:")
    print("-" * 70)

    print("━" * 70)
    print("\nEntry/Exit Chart:")
    print("  • 60-day candlestick chart with key strategy levels")
    print("  • Green line: Recommended entry price")
    print("  • Gold line: First profit target")
    print("  • Blue line: Second profit target")
    print("  • Red line: Stop-loss protection level")
    print("  → Visualizes strategy against recent price action")

    print("\nRisk/Reward Chart:")
    print("  • Bar comparison of potential gain vs. maximum loss")
    print("  • Green bar: Expected profit if targets hit")
    print("  • Red bar: Maximum loss if stop triggered")
    print("  → Quantifies trade asymmetry at a glance")

    print(f"\n📁 Charts saved to: {os.path.join(os.getcwd(), 'charts')}")
    print()


# ======================================================
# MAIN (MARKET FORECASTER STYLE CLI)
# ======================================================
def main():
    print("🔧 Loading configuration...\n")

    print("=" * 70)
    print("📈 PROFESSIONAL TRADE STRATEGY (MARKET FORECASTER STYLE - OLLAMA)")
    print("=" * 70)
    print()

    api_keys = load_all_keys("config_api_keys")

    try:
        ollama_key = load_api_key("config_api_keys")
    except Exception as e:
        print(f"❌ API Key Error: {e}")
        print("   → Make sure OLLAMA_API_KEY is set correctly.")
        return

    try:
        llm_config = setup_llm_config_ollama(
            ollama_api_key=ollama_key,
            model="gpt-oss:20b",
        )
    except Exception as e:
        print(f"❌ LLM Config Error: {e}")
        return

    # --------------------------
    # USER INPUTS
    # --------------------------
    ticker = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
    amount_input = input("Enter Investment Amount (e.g., 10000): ").strip()
    risk = input("Risk Tolerance (Low / Medium / High): ").strip().lower()

    try:
        amount = float(amount_input.replace(",", ""))
        if amount <= 0:
            raise ValueError
    except:
        print("❌ Invalid amount. Enter a positive number.")
        return

    if risk not in ["low", "medium", "high"]:
        print("❌ Risk must be Low, Medium, or High")
        return

    print(f"\n🚀 Starting trade strategy analysis for {ticker}.")

    # --------------------------
    # FETCH DATA
    # --------------------------
    print("\n🔍 Fetching market data...")
    try:
        df = fetch_ohlc(ticker, days=250)
        company_info = get_company_info(ticker)
    except Exception as e:
        print(f"❌ Data Fetch Error: {e}")
        return

    # --------------------------
    # BUILD PROMPT
    # --------------------------
    prompt = build_prompt(ticker, amount, risk, df, company_info)

    # --------------------------
    # CALL LLM
    # --------------------------
    try:
        llm_response = call_autogen(llm_config, prompt)
    except Exception as e:
        print("\n---------------------------------------------------------------")
        print(f"❌ AI Model Error: {e}")
        return

    print("\n📊 Processing strategy...")
    parsed = parse_enhanced_strategy(llm_response)
    parsed = ensure_numbers(parsed, amount, df)

    print("\n📈 Generating charts...")
    charts_dir = ensure_charts_dir()
    charts = {
        "entry_exit": entry_exit_chart(df, parsed, ticker, charts_dir),
        "risk_reward": risk_reward_chart(parsed, ticker, charts_dir),
    }

    # ================================================================
    # 📌 CAPTURE FULL TERMINAL OUTPUT (PROFESSIONAL REPORT)
    # ================================================================
    capture_text = capture_output(
        print_professional_report,
        ticker, amount, risk, parsed, charts, df, company_info, llm_response
    )

    # print normally
    print(capture_text)

    # ================================================================
    # 💾 SAVE FULL OUTPUT TO DATABASE
    # ================================================================
    try:
        log_to_db(
            agent_name="trade_strategy",
            ticker=ticker,
            params={"amount": amount, "risk": risk},
            output=capture_text   # STORE FULL REPORT (not llm_response)
        )
        print("\n💾 Successfully saved trade strategy to database.\n")
    except Exception as db_error:
        print(f"\n⚠️ Warning: Failed to log to database: {db_error}\n")

    print("\n✨ Strategy generation completed.")

if __name__ == "__main__":
    main()