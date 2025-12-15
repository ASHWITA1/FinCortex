import requests
import json
import os
import warnings
import logging
from textwrap import dedent
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

import io
from contextlib import redirect_stdout, redirect_stderr

import os

# =============================
# CHART BASE DIRECTORY (FIX)
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

def capture_output(func, *args, **kwargs):
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            func(*args, **kwargs)
    except Exception as e:
        buffer.write(f"\n[ERROR] {e}\n")
    return buffer.getvalue()


try:
    from config_db import log_to_db
except Exception:
    log_to_db=None


# Suppress warnings and reduce logging noise
warnings.filterwarnings("ignore")
logging.getLogger("autogen").setLevel(logging.ERROR)
logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)

# AutoGen import
try:
    import autogen
except Exception:
    autogen = None
    print("⚠  AutoGen not installed. Install with: pip install pyautogen")

# Try to import yfinance as fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠ yfinance not available. Install with: pip install yfinance")


# ----------------- Load API Keys -----------------
def load_api_keys():
    try:
        with open("config_api_keys", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        try:
            with open("config_api_keys.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            return {}

API_KEYS = load_api_keys()

HF_API_KEY = API_KEYS.get("HF_API_KEY", "")
if not HF_API_KEY or not HF_API_KEY.startswith("hf_"):
    print("❌ Missing or invalid HF_API_KEY")
    raise SystemExit(1)

FINNHUB_KEY = API_KEYS.get("FINNHUB_API_KEY", "")
FMP_KEY = API_KEYS.get("FMP_API_KEY", "")

LLM_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}


# ----------------- Setup AutoGen LLM Config for HuggingFace -----------------
def setup_llm_config_hf(api_key: str, model: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> dict:
    """Configure AutoGen to use HuggingFace API (OpenAI-compatible endpoint)"""
    if not api_key:
        raise ValueError("HF_API_KEY is required to build llm_config.")
    return {
        "config_list": [
            {
                "model": model,
                "api_key": api_key,
                "base_url": "https://router.huggingface.co/v1",
                "api_type": "openai"
            }
        ],
        "timeout": 120,
        "temperature": 0.2,
        "max_tokens": 800
    }


# ----------------- Fetch Stock Quotes -----------------
def get_stock_quote(ticker):
    if FINNHUB_KEY:
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_KEY}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                d = r.json()
                ts = d.get("t")
                ts_fmt = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M UTC") if ts else "N/A"
                return {
                    "price": d.get("c", 0),
                    "high": d.get("h", 0),
                    "low": d.get("l", 0),
                    "previous_close": d.get("pc", 0),
                    "source": "Finnhub",
                    "ts": ts_fmt
                }
        except:
            pass

    if YFINANCE_AVAILABLE:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if not hist.empty:
                return {
                    "price": float(hist["Close"].iloc[-1]),
                    "high": float(hist["High"].iloc[-1]),
                    "low": float(hist["Low"].iloc[-1]),
                    "previous_close": float(hist["Close"].iloc[-2]) if len(hist) > 1 else float(hist["Close"].iloc[-1]),
                    "source": "yfinance",
                    "ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
                }
        except:
            pass

    return {"price": "N/A", "high": "N/A", "low": "N/A", "previous_close": "N/A", "source": "N/A", "ts": "N/A"}


# ----------------- Company Profile -----------------
def get_company_profile(ticker):
    if FMP_KEY:
        try:
            url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_KEY}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                d = r.json()
                if d:
                    p = d[0]
                    return {
                        "name": p.get("companyName", ticker),
                        "sector": p.get("sector", "N/A"),
                        "industry": p.get("industry", "N/A"),
                        "market_cap": p.get("mktCap", "N/A"),
                        "beta": p.get("beta", "N/A")
                    }
        except:
            pass

    if YFINANCE_AVAILABLE:
        info = yf.Ticker(ticker).info or {}
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "beta": info.get("beta", "N/A")
        }

    return {"name": ticker, "sector": "N/A", "industry": "N/A", "market_cap": "N/A", "beta": "N/A"}


# ----------------- Financial Ratios -----------------
def get_financial_ratios(ticker):
    if FMP_KEY:
        try:
            url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?apikey={FMP_KEY}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                d = r.json()
                if d:
                    p = d[0]
                    return {
                        "pe_ratio": p.get("priceEarningsRatio"),
                        "de_ratio": p.get("debtEquityRatio"),
                        "current_ratio": p.get("currentRatio"),
                        "roe": p.get("returnOnEquity"),
                        "profit_margin": p.get("netProfitMargin")
                    }
        except:
            pass

    if YFINANCE_AVAILABLE:
        info = yf.Ticker(ticker).info
        return {
            "pe_ratio": info.get("trailingPE"),
            "de_ratio": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "roe": info.get("returnOnEquity"),
            "profit_margin": info.get("profitMargins")
        }

    return {"pe_ratio": None, "de_ratio": None, "current_ratio": None, "roe": None, "profit_margin": None}


# ----------------- News -----------------
def get_company_news(ticker, days=7):
    items = []

    if FINNHUB_KEY:
        try:
            to_date = datetime.utcnow().date()
            from_date = to_date - timedelta(days=days)
            url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={FINNHUB_KEY}"
            r = requests.get(url, timeout=10).json()
            for n in r[:5]:
                title = n.get("headline", "")
                ts = n.get("datetime", None)
                dt = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else "N/A"
                if title:
                    items.append(f"{dt} — {title}")
            if items:
                return items
        except:
            pass

    if YFINANCE_AVAILABLE:
        try:
            raw = yf.Ticker(ticker).news
            for n in raw[:5]:
                title = n.get("title") or ""
                ts = n.get("providerPublishTime", None)
                dt = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else "N/A"
                items.append(f"{dt} — {title}")
        except:
            pass

    return items if items else ["No recent headlines available"]


# ----------------- Aggregate Data -----------------
def fetch_stock_data_and_news(ticker):
    quote = get_stock_quote(ticker)
    profile = get_company_profile(ticker)
    ratios = get_financial_ratios(ticker)
    news = get_company_news(ticker)

    summary = "No summary available."
    if YFINANCE_AVAILABLE:
        info = yf.Ticker(ticker).info
        summary = info.get("longBusinessSummary", "No summary available.")

    return {
        "name": profile["name"],
        "sector": profile["sector"],
        "industry": profile["industry"],
        "price": quote["price"],
        "price_ts": quote["ts"],
        "price_source": quote["source"],
        "pe_ratio": ratios["pe_ratio"],
        "market_cap": profile["market_cap"],
        "beta": profile["beta"],
        "de_ratio": ratios["de_ratio"],
        "current_ratio": ratios["current_ratio"],
        "roe": ratios["roe"],
        "profit_margin": ratios["profit_margin"],
        "summary": summary,
        "news": news
    }


# ----------------- Prompt Builder -----------------
def build_prompt(data, ticker):
    news_block = "\n".join(f"- {n}" for n in data["news"])
    price_info = f"Price snapshot: {data['price']} (Source: {data['price_source']}, Time: {data['price_ts']})"

    return dedent(f"""
You are a senior financial risk analyst. Follow EXACT format:

OVERALL RISK SCORE (0-10):
- Score: [X]
- Level: MUST match score (0-3: Low Risk, 4-6: Moderate Risk, 7-10: High Risk)
- Meaning: [short explanation]

RISK BREAKDOWN:
- Market Risk: [very low / low / moderate / high]
- Financial Risk: [very low / low / moderate / high]
- Business Risk: [very low / low / moderate / high]

KEY RISK FACTORS:
- [Factor 1]
- [Factor 2]
- [Factor 3]
- [Factor 4]

RISK EXPLANATION:
- Why risky: [explanation]
- What's concerning: [specific risks]
- What's safe: [mitigating factors]
- Meaning: [overall interpretation]

FINAL RECOMMENDATION:
- Action: [BUY / HOLD / SELL]
- Meaning: [short rationale]

USE ONLY THIS DATA:
Ticker: {ticker}
Company: {data['name']}
Sector: {data['sector']}
Industry: {data['industry']}
Price: {data['price']}
P/E Ratio: {data['pe_ratio']}
Market Cap: {data['market_cap']}
Beta: {data['beta']}
Debt-to-Equity: {data['de_ratio']}
Current Ratio: {data['current_ratio']}
ROE: {data['roe']}
Profit Margin: {data['profit_margin']}

Business Summary:
{data['summary']}

RECENT NEWS:
{news_block}

PRICE INFO:
{price_info}
""").strip()


# ----------------- AutoGen Risk Analysis -----------------
def call_autogen_risk_analysis(llm_config: dict, prompt: str) -> tuple:
    if autogen is None:
        print("⚠  AutoGen not available, falling back to direct HF API call...")
        return hf_llm_direct(prompt)
    
    print("🤖 Initializing AutoGen multi-agent risk assessment system...")
    
    risk_analyst = autogen.AssistantAgent(
        name="Risk_Analyst",
        llm_config=llm_config,
        system_message=(
            "You are a senior financial risk analyst with expertise in market risk, "
            "credit risk, and operational risk assessment. You specialize in producing "
            "structured risk reports with precise risk scores and actionable recommendations. "
            "Follow the EXACT format provided. Use ONLY the data given. Be precise and concise."
        )
    )
    
    user_proxy = autogen.UserProxyAgent(
        name="Risk_Manager",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False
    )
    
    print("🤖 Running comprehensive risk analysis via AutoGen agents...")
    
    try:
        user_proxy.initiate_chat(
            risk_analyst,
            message=prompt,
            clear_history=True
        )
        
        last_message = user_proxy.last_message()
        if not last_message:
            raise RuntimeError("AutoGen agents did not return a response.")
        
        result = last_message.get("content", "").strip()
        return True, result
    
    except Exception as e:
        print(f"⚠  AutoGen error: {e}")
        print("Falling back to direct API call...")
        return hf_llm_direct(prompt)


# ----------------- Fallback: Direct HF LLM -----------------
def hf_llm_direct(prompt):
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
            {"role": "system", "content": "Strict-format financial analyst."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 800
    }

    r = requests.post(LLM_URL, headers=HEADERS, json=payload)
    if r.status_code != 200:
        return False, r.text

    try:
        return True, r.json()["choices"][0]["message"]["content"]
    except:
        return False, r.text


# ----------------- Parse LLM output -----------------
def parse_llm_output(text):
    score = None
    level = ""
    meaning = ""
    market = financial = business = None
    risk_factors = []
    why_risky = ""
    whats_concerning = ""
    whats_safe = ""
    risk_meaning = ""
    recommendation = ""
    rec_meaning = ""

    lines = text.splitlines()
    current_section = None
    
    for line in lines:
        line_strip = line.strip()
        
        # Section detection
        if "OVERALL RISK SCORE" in line_strip.upper():
            current_section = "score"
        elif "RISK BREAKDOWN" in line_strip.upper():
            current_section = "breakdown"
        elif "KEY RISK FACTORS" in line_strip.upper():
            current_section = "factors"
        elif "RISK EXPLANATION" in line_strip.upper():
            current_section = "explanation"
        elif "FINAL RECOMMENDATION" in line_strip.upper():
            current_section = "recommendation"
        
        # Parse based on section
        if current_section == "score":
            if line_strip.startswith("- Score:"):
                try:
                    score = int(line_strip.split(":")[1].strip())
                except:
                    pass
            elif line_strip.startswith("- Level:"):
                level = line_strip.split(":", 1)[1].strip()
            elif line_strip.startswith("- Meaning:"):
                meaning = line_strip.split(":", 1)[1].strip()
        
        elif current_section == "breakdown":
            if line_strip.startswith("- Market Risk:"):
                market = line_strip.split(":", 1)[1].strip().lower()
            elif line_strip.startswith("- Financial Risk:"):
                financial = line_strip.split(":", 1)[1].strip().lower()
            elif line_strip.startswith("- Business Risk:"):
                business = line_strip.split(":", 1)[1].strip().lower()
        
        elif current_section == "factors":
            if line_strip.startswith("-") and len(line_strip) > 2:
                risk_factors.append(line_strip[1:].strip())
        
        elif current_section == "explanation":
            if line_strip.startswith("- Why risky:"):
                why_risky = line_strip.split(":", 1)[1].strip()
            elif line_strip.startswith("- What's concerning:"):
                whats_concerning = line_strip.split(":", 1)[1].strip()
            elif line_strip.startswith("- What's safe:"):
                whats_safe = line_strip.split(":", 1)[1].strip()
            elif line_strip.startswith("- Meaning:"):
                risk_meaning = line_strip.split(":", 1)[1].strip()
        
        elif current_section == "recommendation":
            if line_strip.startswith("- Action:"):
                recommendation = line_strip.split(":", 1)[1].strip()
            elif line_strip.startswith("- Meaning:"):
                rec_meaning = line_strip.split(":", 1)[1].strip()

    return {
        "score": score,
        "level": level,
        "meaning": meaning,
        "market": market,
        "financial": financial,
        "business": business,
        "risk_factors": risk_factors,
        "why_risky": why_risky,
        "whats_concerning": whats_concerning,
        "whats_safe": whats_safe,
        "risk_meaning": risk_meaning,
        "recommendation": recommendation,
        "rec_meaning": rec_meaning
    }


# ----------------- Label → numeric -----------------
def convert_label_to_val(label):
    table = {"very low": 1, "low": 2, "moderate": 3, "high": 4}
    return table.get(label, 2)


# ----------------- Enhanced Output Display -----------------
def display_enhanced_risk_report(ticker, data, parsed, ist_time):
    """Display risk assessment in portfolio-style format with detailed explanations"""
    
    score = parsed["score"]
    level = parsed["level"]
    
    # Determine risk color
    if score is not None:
        if score <= 3:
            risk_emoji = "🟢"
            risk_color = "Low Risk"
        elif score <= 6:
            risk_emoji = "🟡"
            risk_color = "Moderate Risk"
        else:
            risk_emoji = "🔴"
            risk_color = "High Risk"
    else:
        risk_emoji = "⚪"
        risk_color = "Unknown"
    
    print("\n" + "=" * 70)
    print("🛡  COMPREHENSIVE RISK ASSESSMENT REPORT")
    
    # Overall Risk Score
    print("\n📌 Overall Risk Assessment:")
    print("-" * 70)

    print("━" * 70)
    if score is not None:
        print(f"   {risk_emoji} Risk Score: {score}/10 ({risk_color})")
        print(f"   Risk Level: {level}")
        if parsed["meaning"]:
            print(f"   Interpretation: {parsed['meaning']}")
    else:
        print(f"   ⚪ Risk Score: Unable to calculate")
    
    print(f"\n   💬 Understanding Risk Scores:")
    print(f"      • 0-3: Low Risk → Suitable for conservative investors")
    print(f"      • 4-6: Moderate Risk → Balanced risk-reward profile")
    print(f"      • 7-10: High Risk → Only for aggressive risk-takers")
    print(f"      Higher scores indicate greater volatility and uncertainty")
    
    # Risk Breakdown
    print("\n📌 Risk Breakdown Analysis:")
    print("-" * 70)

    print("━" * 70)
    
    m_risk = parsed["market"]
    f_risk = parsed["financial"]
    b_risk = parsed["business"]
    
    if m_risk:
        m_emoji = "🟢" if "low" in m_risk else "🟡" if "moderate" in m_risk else "🔴"
        print(f"   {m_emoji} Market Risk: {m_risk.title()}")
    if f_risk:
        f_emoji = "🟢" if "low" in f_risk else "🟡" if "moderate" in f_risk else "🔴"
        print(f"   {f_emoji} Financial Risk: {f_risk.title()}")
    if b_risk:
        b_emoji = "🟢" if "low" in b_risk else "🟡" if "moderate" in b_risk else "🔴"
        print(f"   {b_emoji} Business Risk: {b_risk.title()}")
    
    print(f"\n   💬 Risk Category Definitions:")
    print("-" * 70)

    print(f"      • Market Risk: Exposure to overall market movements, economic")
    print(f"        conditions, and sector-specific trends. High beta stocks have")
    print(f"        higher market risk.")
    print(f"      • Financial Risk: Company's debt levels, liquidity, profitability,")
    print(f"        and ability to meet obligations. Poor ratios = higher risk.")
    print(f"      • Business Risk: Industry competition, regulatory changes, product")
    print(f"        demand, and company-specific operational challenges.")
    
    # Key Risk Factors
    print("\n📌 Key Risk Factors:")
    print("-" * 70)

    print("━" * 70)
    if parsed["risk_factors"]:
        for i, factor in enumerate(parsed["risk_factors"], 1):
            print(f"  {i}. ⚠  {factor}")
    else:
        print("  ℹ  No specific risk factors identified in analysis")
    
    print(f"\n   💬 How to Use Risk Factors:")
    print(f"      These are specific concerns that could negatively impact the stock.")
    print(f"      Monitor these factors closely - if they worsen, consider reducing")
    print(f"      exposure. Use them to set alerts and track ongoing developments.")
    
    # Detailed Risk Explanation
    print("\n📌 Detailed Risk Analysis:")
    print("-" * 70)

    print("━" * 70)
    if parsed["why_risky"]:
        print(f"   🔍 Why This Stock Is Risky:")
        print(f"      {parsed['why_risky']}")
    
    if parsed["whats_concerning"]:
        print(f"\n   ⚠  Primary Concerns:")
        print(f"      {parsed['whats_concerning']}")
    
    if parsed["whats_safe"]:
        print(f"\n   ✅ Mitigating Factors:")
        print(f"      {parsed['whats_safe']}")
    
    if parsed["risk_meaning"]:
        print(f"\n   💬 Overall Interpretation:")
        print(f"      {parsed['risk_meaning']}")
    
    print(f"\n   💬 Balanced Perspective:")
    print(f"      Every investment has risks AND opportunities. The key is matching")
    print(f"      your risk tolerance with the stock's risk profile. Higher risk can")
    print(f"      mean higher returns, but also higher potential losses.")
    
    # Financial Health Snapshot
    print("\n📌 Financial Health Snapshot:")
    print("-" * 70)

    print("━" * 70)
    print(f"   Current Price: ${data['price']}")
    print(f"   Market Cap: {data['market_cap']:,}" if isinstance(data['market_cap'], (int, float)) else f"   Market Cap: {data['market_cap']}")
    print(f"   P/E Ratio: {data['pe_ratio'] if data['pe_ratio'] else 'N/A'}")
    print(f"   Beta: {data['beta'] if data['beta'] else 'N/A'}")
    print(f"   Debt-to-Equity: {data['de_ratio'] if data['de_ratio'] else 'N/A'}")
    print(f"   Current Ratio: {data['current_ratio'] if data['current_ratio'] else 'N/A'}")
    print(f"   ROE: {data['roe'] if data['roe'] else 'N/A'}")
    print(f"   Profit Margin: {data['profit_margin'] if data['profit_margin'] else 'N/A'}")
    
    print(f"\n   💬 Financial Metrics Guide:")
    print(f"      • Beta > 1.0: More volatile than market (higher market risk)")
    print(f"      • Debt-to-Equity > 1.0: Leverage risk (higher financial risk)")
    print(f"      • Current Ratio < 1.5: Liquidity concerns")
    print(f"      • Low ROE: Poor profitability (business risk)")
    
    # Investment Recommendation
    print("\n📌 Investment Recommendation:")
    print("-" * 70)

    print("━" * 70)
    rec = parsed["recommendation"].upper()
    if "BUY" in rec:
        rec_emoji = "✅"
        rec_action = "BUY"
    elif "SELL" in rec:
        rec_emoji = "🔴"
        rec_action = "SELL"
    else:
        rec_emoji = "⏸"
        rec_action = "HOLD"
    
    print(f"   {rec_emoji} Recommendation: {rec_action}")
    if parsed["rec_meaning"]:
        print(f"   Rationale: {parsed['rec_meaning']}")
    
    print(f"\n   💬 Recommendation Context:")
    if rec_action == "BUY":
        print(f"      • BUY signal suggests risk-reward is favorable at current price")
        print(f"      • Consider position sizing based on your risk tolerance")
        print(f"      • Set stop-losses to protect against downside")
    elif rec_action == "SELL":
        print(f"      • SELL signal suggests risks outweigh potential rewards")
        print(f"      • Consider reducing exposure or exiting position")
        print(f"      • Look for better risk-adjusted opportunities")
    else:
        print(f"      • HOLD signal suggests waiting for clearer signals")
        print(f"      • Monitor for changes in risk factors")
        print(f"      • Neither compelling buy nor urgent sell at this time")
    
    # Risk Management Strategies
    print("\n📌 Risk Management Strategies:")
    print("-" * 70)

    print("━" * 70)
    print(f"  1. 📊 Position Sizing:")
    if score is not None:
        if score <= 3:
            print(f"     Low risk allows for larger position (5-10% of portfolio)")
        elif score <= 6:
            print(f"     Moderate risk suggests medium position (3-5% of portfolio)")
        else:
            print(f"     High risk requires smaller position (1-3% of portfolio)")
    
    print(f"\n  2. 🛡  Stop-Loss Strategy:")
    if data['price'] and isinstance(data['price'], (int, float)):
        stop_5 = data['price'] * 0.95
        stop_10 = data['price'] * 0.90
        print(f"     Set stop-loss at ${stop_5:.2f} (-5%) or ${stop_10:.2f} (-10%)")
    else:
        print(f"     Set stop-loss 5-10% below entry price")
    
    print(f"\n  3. 🔄 Monitoring Plan:")
    print(f"     • Review risk factors weekly")
    print(f"     • Set price alerts at key support/resistance levels")
    print(f"     • Track news for changes in business/financial risk")
    print(f"     • Rebalance if risk profile changes significantly")
    
    print(f"\n  4. 💼 Diversification:")
    print(f"     • Don't put all eggs in one basket")
    print(f"     • Balance with lower-risk assets")
    print(f"     • Consider sector and geographic diversification")

    
    # Charts Generated
    print("\n📊 Visual Analytics Generated:")
    print("-" * 70)

    print("━" * 70)
    print(f"  ✅ Risk Gauge Chart: ./charts/risk_gauge_{ticker}.png")
    print(f"  ✅ Risk Breakdown Pie Chart: ./charts/risk_breakdown_{ticker}.png")
    print(f"  ✅ 6-Month Volatility Chart: ./charts/volatility_{ticker}.png")
    

    
    print("\n" + "=" * 70)
    print("✨ Risk Assessment completed successfully.")
    print("=" * 70)

# ----------------- SAVE CHARTS -----------------
def plot_risk_gauge(score, ticker):
    os.makedirs("charts", exist_ok=True)
    filepath = os.path.join(CHART_DIR, f"risk_gauge_{ticker}.png")


    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)

    color = "green" if score <= 3 else "yellow" if score <= 6 else "red"
    ax.barh(0.5, score, color=color, height=0.4)
    ax.barh(0.5, 10, color="lightgray", alpha=0.3, height=0.4)

    ax.text(score + 0.3, 0.5, f"{score}/10", va="center", fontsize=16, fontweight='bold')
    
    # Add risk level text
    if score <= 3:
        risk_text = "LOW RISK"
    elif score <= 6:
        risk_text = "MODERATE RISK"
    else:
        risk_text = "HIGH RISK"
    
    ax.text(5, 0.1, risk_text, va="center", ha="center", fontsize=12, style='italic')
    ax.set_title("Overall Risk Score Gauge", fontsize=14, fontweight='bold')
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✔ Saved Risk Gauge → {filepath}")


def plot_risk_pie(m, f, b, ticker):
    os.makedirs("charts", exist_ok=True)
    filepath = os.path.join(CHART_DIR, f"risk_breakdown_{ticker}.png")

    labels = ["Market Risk", "Financial Risk", "Business Risk"]
    values = [m, f, b]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.05, 0.05, 0.05)

    plt.figure(figsize=(8, 6))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, 
            colors=colors, explode=explode, shadow=True)
    plt.title("Risk Category Breakdown", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✔ Saved Risk Breakdown Pie → {filepath}")


def plot_volatility(ticker):
    os.makedirs("charts", exist_ok=True)
    filepath = os.path.join(CHART_DIR, f"volatility_{ticker}.png")

    if not YFINANCE_AVAILABLE:
        print("⚠ yfinance missing - cannot generate volatility chart")
        return

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="6mo")
        if hist.empty:
            print("⚠ No historical data available for volatility chart")
            return

        hist["Returns"] = hist["Close"].pct_change() * 100  # Convert to percentage

        plt.figure(figsize=(12, 5))
        
        # Plot returns
        plt.plot(hist.index, hist["Returns"], linewidth=1, alpha=0.7, color='steelblue')
        plt.axhline(0, color="black", linewidth=0.8, linestyle='--')
        
        # Add standard deviation bands
        std = hist["Returns"].std()
        plt.axhline(std, color="red", linewidth=0.8, linestyle=':', alpha=0.5, label=f'+1 Std Dev ({std:.2f}%)')
        plt.axhline(-std, color="red", linewidth=0.8, linestyle=':', alpha=0.5, label=f'-1 Std Dev ({-std:.2f}%)')
        
        plt.title(f"6-Month Price Volatility Analysis – {ticker}", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=10)
        plt.ylabel("Daily Returns (%)", fontsize=10)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✔ Saved Volatility Chart → {filepath}")
    except Exception as e:
        print(f"⚠ Error generating volatility chart: {e}")

# =============================
# DATABASE SUPPORT
# =============================
# try:
#     from config_db import log_to_db
# except Exception:
#     log_to_db = None


# warnings.filterwarnings("ignore")
# logging.getLogger("autogen").setLevel(logging.ERROR)
# logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)

# try:
#     import autogen
# except Exception:
#     autogen = None
#     print("⚠  AutoGen not installed. Install with: pip install pyautogen")

# try:
#     import yfinance as yf
#     YFINANCE_AVAILABLE = True
# except ImportError:
#     YFINANCE_AVAILABLE = False
#     print("⚠ yfinance not available. Install with: pip install yfinance")


# # ----------------- Load API Keys -----------------
# def load_api_keys():
#     try:
#         with open("config_api_keys", "r", encoding="utf-8") as f:
#             return json.load(f)
#     except:
#         try:
#             with open("config_api_keys.json", "r", encoding="utf-8") as f:
#                 return json.load(f)
#         except Exception as e:
#             print(f"❌ Error loading config file: {e}")
#             return {}


# API_KEYS = load_api_keys()

# HF_API_KEY = API_KEYS.get("HF_API_KEY", "")
# if not HF_API_KEY or not HF_API_KEY.startswith("hf_"):
#     print("❌ Missing or invalid HF_API_KEY")
#     raise SystemExit(1)

# FINNHUB_KEY = API_KEYS.get("FINNHUB_API_KEY", "")
# FMP_KEY = API_KEYS.get("FMP_API_KEY", "")

# LLM_URL = "https://router.huggingface.co/v1/chat/completions"
# HEADERS = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}


# # ----------------- Setup AutoGen LLM Config for HuggingFace -----------------
# def setup_llm_config_hf(api_key: str, model: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> dict:
#     if not api_key:
#         raise ValueError("HF_API_KEY is required to build llm_config.")
#     return {
#         "config_list": [
#             {
#                 "model": model,
#                 "api_key": api_key,
#                 "base_url": "https://router.huggingface.co/v1",
#                 "api_type": "openai"
#             }
#         ],
#         "timeout": 120,
#         "temperature": 0.2,
#         "max_tokens": 800
#     }
# # =============================
# # MAIN SCRIPT (DB integrated)
# # =============================
def main():
    print("\n" + "=" * 70)
    print("🛡  ENHANCED RISK ASSESSMENT TOOL")
    print("=" * 70 + "\n")

    if autogen:
        print("✓ AutoGen framework detected - using multi-agent risk analysis")
    else:
        print("⚠ AutoGen not installed - using direct API mode")
        print("  Install with: pip install pyautogen\n")

    # -----------------------------
    # USER INPUT
    # -----------------------------
    ticker = input("Enter stock ticker (e.g., AAPL, TSLA, TCS.NS): ").upper().strip()
    if not ticker:
        print("❌ Please enter a ticker symbol")
        return

    print("\n📊 Fetching comprehensive market data...")
    data = fetch_stock_data_and_news(ticker)

    # Current IST time
    ist = datetime.utcnow() + timedelta(hours=5, minutes=30)

    # LLM config
    try:
        llm_config = setup_llm_config_hf(api_key=HF_API_KEY)
    except Exception as e:
        print(f"❌ LLM Config Error: {e}")
        return

    print("🤖 Generating comprehensive risk analysis via AI...\n")

    # Build prompt for LLM
    prompt = build_prompt(data, ticker)

    # Run analysis via AutoGen or fallback
    ok, result = call_autogen_risk_analysis(llm_config, prompt)
    if not ok:
        print("❌ LLM Error:", result)
        return

    # Parse LLM Output
    parsed = parse_llm_output(result)

    # ---------------------------------------------------------
    # CAPTURE FULL TERMINAL OUTPUT FROM REPORT
    # ---------------------------------------------------------
    full_output = capture_output(
        display_enhanced_risk_report,
        ticker, data, parsed, ist
    )

    # Print to terminal
    print(full_output)

    # ---------------------------------------------------------
    # CHART GENERATION
    # ---------------------------------------------------------
    score = parsed["score"]
    if score is None:
        print("\n⚠ Could not parse risk score. Charts skipped.")
    else:
        print("\n📊 Generating visual analytics...\n")

        m_val = convert_label_to_val(parsed["market"])
        f_val = convert_label_to_val(parsed["financial"])
        b_val = convert_label_to_val(parsed["business"])

        plot_risk_gauge(score, ticker)
        plot_risk_pie(m_val, f_val, b_val, ticker)
        plot_volatility(ticker)

        print("\n✅ All charts saved to ./charts/ directory")
        print("=" * 70 + "\n")

    # ---------------------------------------------------------
    # SAVE FULL OUTPUT TO DATABASE
    # ---------------------------------------------------------
    if log_to_db:
        try:
            log_to_db(
                agent_name="risk_assessment",
                ticker=ticker,
                params={},
                output=full_output   # FULL TERMINAL OUTPUT STORED HERE
            )
            print("\n💾 Successfully saved risk assessment to database.\n")
        except Exception as e:
            print(f"\n⚠ Warning: Failed to log to database: {e}\n")
    else:
        print("\n⚠ config_db not found — skipping DB logging.\n")


# ----------------- ENTRY -----------------
if __name__ == "__main__":
    main()