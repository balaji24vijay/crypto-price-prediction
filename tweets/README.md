# CryptoVault

CryptoVault is a Flask dashboard for short-horizon cryptocurrency estimates and conversation sentiment analysis.

## What was upgraded

- Removed embedded API and SMTP credentials.
- Added environment-based configuration and safer session cookies.
- Added password hashing with transparent migration for older plain-text records.
- Added input validation, duplicate-account checks, authenticated dashboard routes, and a health endpoint.
- Replaced the fragile TensorFlow-only path with a lightweight volatility-damped momentum baseline that works on ordinary machines.
- Added robust market-data error handling, cached CSV history, clearer forecast charts, and an optional JSON forecast endpoint.
- Added an offline sentiment fallback when the NLTK VADER corpus is unavailable.
- Removed automatic email delivery from every forecast request.

## Run locally

1. Create a virtual environment and install dependencies:

   py -m venv .venv
   .venv/Scripts/Activate.ps1
   pip install -r requirements.txt

2. Copy .env.example to .env and set FLASK_SECRET_KEY. A CryptoCompare API key is recommended for reliable live data.

3. Start the app:

   flask --app app run

Open http://127.0.0.1:5000. Create an account, then select up to five coins.

## Routes

- / sign in
- /logon create an account
- /next forecast dashboard
- /sen sentiment dashboard
- /api/forecast/BTC authenticated JSON forecast
- /healthz health check

Forecasts are educational estimates and should not be used as the sole basis for financial decisions.
