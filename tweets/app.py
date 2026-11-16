from __future__ import annotations

import hmac
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from email.utils import parseaddr
from functools import wraps
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
STATIC_DIR = BASE_DIR / "static"
GRAPH_DIR = STATIC_DIR / "graphs"
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(BASE_DIR / "signup.db")))
DATASET_PATH = Path(os.getenv("SENTIMENT_DATASET", str(BASE_DIR / "crypto.csv")))
if not DATASET_PATH.exists():
    DATASET_PATH = BASE_DIR.parent / "crypto.csv"
if not DATASET_PATH.exists():
    DATASET_PATH = BASE_DIR.parent.parent / "crypto.csv"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

COINS = ("BTC", "ETH", "XRP", "ADA", "DOGE", "SOL", "BNB", "LTC", "DOT", "MATIC")
MAX_COINS_PER_REQUEST = 5
PREDICTION_HORIZON = max(1, int(os.getenv("PREDICTION_HORIZON", "12")))
REQUEST_TIMEOUT = max(3, int(os.getenv("REQUEST_TIMEOUT", "15")))
CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/histohour"
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "").strip()

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "0") == "1",
)


def get_db() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DATABASE_PATH))
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def db_session():
    connection = get_db()
    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with db_session() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                mobile TEXT NOT NULL,
                name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
            )
            """
        )


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "username" not in session:
            flash("Please sign in to access your dashboard.", "warning")
            return redirect(url_for("index", next=request.path))
        return view(*args, **kwargs)

    return wrapped_view


def valid_email(value: str) -> bool:
    return bool(parseaddr(value)[1] and re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value))


def password_matches(stored_password: str, candidate: str) -> bool:
    if stored_password.startswith(("scrypt:", "pbkdf2:")):
        return check_password_hash(stored_password, candidate)
    # Older project databases stored plain text passwords. Accept them once,
    # then the login route transparently upgrades the record to a hash.
    return hmac.compare_digest(stored_password, candidate)


def get_crypto_data(symbol: str) -> pd.DataFrame:
    symbol = symbol.upper().strip()
    if symbol not in COINS:
        raise ValueError("That cryptocurrency is not supported.")

    params = {"fsym": symbol, "tsym": "USD", "limit": 240}
    if CRYPTOCOMPARE_API_KEY:
        params["api_key"] = CRYPTOCOMPARE_API_KEY

    response = requests.get(CRYPTOCOMPARE_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    if payload.get("Response") == "Error":
        raise RuntimeError(payload.get("Message", "The market-data provider returned an error."))

    rows = (payload.get("Data") or {}).get("Data") or []
    frame = pd.DataFrame(rows)
    if "close" not in frame:
        raise RuntimeError("The market-data provider returned no price history.")

    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"])
    frame = frame[frame["close"] > 0].reset_index(drop=True)
    if len(frame) < 24:
        raise RuntimeError("Not enough recent price history to create a forecast.")

    frame.to_csv(DATA_DIR / f"{symbol}.csv", index=False)
    return frame[["close"]]


def forecast_prices(data: pd.DataFrame, horizon: int = PREDICTION_HORIZON) -> list[float]:
    """Create a stable short-horizon momentum forecast.

    This keeps the dashboard usable on normal machines without requiring a
    heavyweight deep-learning runtime. It uses recent log-return momentum with
    volatility damping and never predicts a non-positive price.
    """

    values = pd.to_numeric(data["close"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 24:
        raise ValueError("At least 24 prices are required for a forecast.")

    recent = values[-min(72, len(values)) :]
    log_returns = np.diff(np.log(recent))
    trend = float(np.mean(log_returns[-24:]))
    volatility = float(np.std(log_returns[-24:]))
    damping = 1.0 / (1.0 + (volatility * 8.0))
    drift = np.clip(trend * damping, -0.03, 0.03)

    steps = np.arange(1, horizon + 1, dtype=float)
    predictions = recent[-1] * np.exp(drift * steps)
    return np.maximum(predictions, 0.00000001).round(8).tolist()


def create_forecast_chart(symbol: str, data: pd.DataFrame, predictions: list[float]) -> str:
    values = data["close"].to_numpy(dtype=float)
    history_start = max(0, len(values) - 96)
    history = values[history_start:]
    history_x = np.arange(history_start, len(values))
    future_x = np.arange(len(values), len(values) + len(predictions))

    plt.style.use("dark_background")
    figure, axis = plt.subplots(figsize=(11, 4.8))
    figure.patch.set_facecolor("#090b14")
    axis.set_facecolor("#111522")
    axis.plot(history_x, history, color="#8b7cff", linewidth=2.2, label="Recent price")
    axis.plot(
        future_x,
        predictions,
        color="#35d6b2",
        linewidth=2.2,
        linestyle="--",
        label=f"{len(predictions)}-hour forecast",
    )
    axis.fill_between(future_x, predictions, alpha=0.08, color="#35d6b2")
    axis.set_title(f"{symbol} / USD forecast", loc="left", color="#f5f7ff", pad=14)
    axis.set_xlabel("Hourly observations", color="#8f99ad")
    axis.set_ylabel("Price (USD)", color="#8f99ad")
    axis.grid(alpha=0.12)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()

    output_path = GRAPH_DIR / f"{symbol}.png"
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return url_for("static", filename=f"graphs/{symbol}.png")


def build_forecast(symbol: str) -> dict[str, Any]:
    data = get_crypto_data(symbol)
    predictions = forecast_prices(data)
    current = float(data["close"].iloc[-1])
    next_price = float(predictions[0])
    change = ((next_price - current) / current) * 100
    return {
        "symbol": symbol,
        "current": round(current, 8),
        "next_price": round(next_price, 8),
        "change": round(change, 2),
        "low": round(min(predictions), 8),
        "high": round(max(predictions), 8),
        "predictions": predictions,
        "graph": create_forecast_chart(symbol, data, predictions),
        "model": "Momentum baseline",
    }


def sentiment_label(text: Any, analyzer: Any) -> str:
    if pd.isna(text):
        return "neutral"
    if analyzer is not None:
        compound = analyzer.polarity_scores(str(text))["compound"]
        if compound >= 0.05:
            return "positive"
        if compound <= -0.05:
            return "negative"
        return "neutral"

    # Small fallback lexicon for offline use when the NLTK VADER corpus is not installed.
    positive_words = {"bullish", "buy", "growth", "gain", "moon", "profit", "strong", "up"}
    negative_words = {"bearish", "sell", "loss", "crash", "weak", "down", "risk"}
    words = set(re.findall(r"[a-z]+", str(text).lower()))
    positive = len(words & positive_words)
    negative = len(words & negative_words)
    if positive > negative:
        return "positive"
    if negative > positive:
        return "negative"
    return "neutral"


def analyze_sentiment() -> dict[str, Any]:
    columns = ("Date", "Content")
    if DATASET_PATH.exists():
        frame = pd.read_csv(DATASET_PATH, usecols=lambda column: column in columns)
    else:
        frame = pd.DataFrame(columns=columns)
    if "Content" not in frame:
        frame["Content"] = ""

    analyzer = None
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
    except Exception:
        pass

    frame["Sentiment"] = frame["Content"].apply(lambda value: sentiment_label(value, analyzer))
    counts = frame["Sentiment"].value_counts().to_dict()
    positive = int(counts.get("positive", 0))
    negative = int(counts.get("negative", 0))
    neutral = int(counts.get("neutral", 0))
    total = positive + negative + neutral
    percentages = {
        "positive": round(positive / total * 100, 2) if total else 0,
        "negative": round(negative / total * 100, 2) if total else 0,
        "neutral": round(neutral / total * 100, 2) if total else 0,
    }

    plt.style.use("dark_background")
    figure, axis = plt.subplots(figsize=(10, 4.5))
    figure.patch.set_facecolor("#090b14")
    axis.set_facecolor("#111522")
    labels = ["Positive", "Neutral", "Negative"]
    values = [positive, neutral, negative]
    bars = axis.bar(labels, values, color=["#35d6b2", "#8f99ad", "#ff6b83"], width=0.58)
    axis.set_title("Market conversation sentiment", loc="left", color="#f5f7ff", pad=14)
    axis.set_ylabel("Posts", color="#8f99ad")
    axis.grid(axis="y", alpha=0.12)
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            color="#f5f7ff",
        )
    figure.tight_layout()
    figure.savefig(STATIC_DIR / "chart.png", dpi=140, bbox_inches="tight")
    plt.close(figure)

    mood = "Bullish" if positive > negative else "Bearish" if negative > positive else "Balanced"
    return {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "percentages": percentages,
        "total": total,
        "mood": mood,
    }


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "service": "crypto-dashboard"})


@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/logon")
@app.route("/signup", methods=["GET"])
def signup_page():
    return render_template("signup.html")


@app.route("/signup", methods=["POST"])
def signup():
    username = request.form.get("user", "").strip()
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    mobile = request.form.get("mobile", "").strip()
    password = request.form.get("password", "")

    if len(username) < 3 or len(name) < 2:
        flash("Please enter a valid username and full name.", "error")
        return render_template("signup.html"), 400
    if not valid_email(email):
        flash("Please enter a valid email address.", "error")
        return render_template("signup.html"), 400
    if not re.fullmatch(r"[0-9+() -]{7,20}", mobile):
        flash("Please enter a valid mobile number.", "error")
        return render_template("signup.html"), 400
    if len(password) < 8:
        flash("Your password must be at least 8 characters.", "error")
        return render_template("signup.html"), 400

    try:
        with db_session() as connection:
            existing = connection.execute(
                'SELECT 1 FROM info WHERE "user" = ? OR email = ?',
                (username, email),
            ).fetchone()
            if existing:
                flash("That username or email is already registered.", "error")
                return render_template("signup.html"), 409
            connection.execute(
                """
                INSERT INTO info ("user", email, password, mobile, name, role)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (username, email, generate_password_hash(password), mobile, name, "user"),
            )
    except sqlite3.Error:
        flash("The account could not be created right now. Please try again.", "error")
        return render_template("signup.html"), 500

    flash("Account created. You can sign in now.", "success")
    return redirect(url_for("index"))


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return redirect(url_for("index"))

    username = request.form.get("user", "").strip()
    password = request.form.get("password", "")
    with db_session() as connection:
        account = connection.execute(
            'SELECT "user", password, mobile, name FROM info WHERE "user" = ?',
            (username,),
        ).fetchone()

        if not account or not password_matches(account["password"], password):
            flash("Invalid username or password.", "error")
            return redirect(url_for("index"))

        if not account["password"].startswith(("scrypt:", "pbkdf2:")):
            connection.execute(
                'UPDATE info SET password = ? WHERE "user" = ?',
                (generate_password_hash(password), username),
            )

    session.clear()
    session["username"] = account["user"]
    session["name"] = account["name"]
    session["mobile"] = account["mobile"]
    return redirect(url_for("dashboard"))


@app.route("/forgot")
def forgot():
    flash("Password reset is not configured yet. Please contact the administrator.", "warning")
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("index"))


@app.route("/next", methods=["GET", "POST"])
@login_required
def dashboard():
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    selected = [coin for coin in request.form.getlist("coins") if coin in COINS]

    if request.method == "POST":
        if not selected:
            flash("Choose at least one cryptocurrency to start a forecast.", "warning")
        else:
            if len(selected) > MAX_COINS_PER_REQUEST:
                selected = selected[:MAX_COINS_PER_REQUEST]
                flash(f"Only the first {MAX_COINS_PER_REQUEST} selections were processed.", "warning")
            for coin in selected:
                try:
                    results.append(build_forecast(coin))
                except (requests.RequestException, ValueError, RuntimeError) as error:
                    errors.append(f"{coin}: {error}")

    return render_template(
        "next.html",
        coins=COINS,
        selected=selected,
        results=results,
        errors=errors,
        username=session.get("name") or session.get("username"),
        horizon=PREDICTION_HORIZON,
    )


@app.route("/api/forecast/<symbol>")
@login_required
def forecast_api(symbol: str):
    try:
        return jsonify(build_forecast(symbol.upper()))
    except (requests.RequestException, ValueError, RuntimeError) as error:
        return jsonify({"error": str(error)}), 400


@app.route("/sen")
@login_required
def sentiment():
    summary = analyze_sentiment()
    return render_template(
        "senindex.html",
        **summary,
        chart=url_for("static", filename="chart.png"),
    )


init_db()

if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
