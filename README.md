# 📈 Advanced Quant Finance & Asset Pricing Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Finance](https://img.shields.io/badge/Domain-Quantitative_Finance-00C853)

An institutional-grade portfolio analysis tool built for modern quantitative analysts. This dashboard demonstrates advanced Data Engineering (ETL resiliency, API handling) alongside stochastic mathematical modeling and asset pricing.

## 🚀 Features

### 1. Quantitative Finance Models
- **Monte Carlo Price Path Simulation**: Generates 100+ future price trajectories using Geometric Brownian Motion (GBM) based on historical drift and volatility.
- **Value at Risk (VaR)**: Calculates historical VaR at configurable confidence intervals (90%, 95%, 99%) to estimate maximum expected portfolio drawdown.
- **Black-Scholes Options Pricing**: A live theoretical pricing calculator for European Call options with dynamic inputs for Spot, Strike, Time to Maturity, Risk-Free Rate, and Implied Volatility.

### 2. Resilient Data Pipeline (ETL)
- **Live API Fetching**: Integrates with Yahoo Finance (`yfinance`).
- **Fail-Safe Mechanism**: Includes a robust fallback mechanism. If the live API is restricted or rate-limited (HTTP 429), the engine automatically falls back to generating stochastic Mock Data to keep the UI functional.
- **Custom CSV Injection**: Analysts can drag-and-drop their own historical `.csv` files for isolated, offline analysis.

### 3. Premium UI/UX
- **Glassmorphism Design**: Custom CSS injection featuring a dark premium theme, blur filters, and micro-animations.
- **Interactive Visualizations**: High-performance Plotly charts (Candlesticks, Bollinger Bands, RSI trends) mapped to dark mode templates.

## 🛠️ Stack & Installation

**Tech Stack**: Streamlit, Pandas, NumPy, SciPy (Stats), Plotly, Scikit-Learn.

### Local Setup
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd stock_dashboard
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the engine:
   ```bash
   streamlit run app.py
   ```

## 👨‍💻 Author
Built by **[Votre Nom]** - M.Sc. Mathematical & Computational Finance.
Feel free to connect with me on [LinkedIn](#).
