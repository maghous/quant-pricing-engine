import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import analytics

# --- UI CONFIGURATION & CUSTOM CSS ---
st.set_page_config(page_title="Quant Finance Dashboard", layout="wide", initial_sidebar_state="expanded")

# Inject Premium Custom CSS (Glassmorphism & Modern Dark Theme)
st.markdown("""
    <style>
        /* Main background and text */
        .stApp {
            background-color: #0E1117;
            font-family: 'Inter', sans-serif;
        }
        
        /* Metric Cards Styling (Glassmorphism effect) */
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(0, 255, 128, 0.15);
            border: 1px solid rgba(0, 255, 128, 0.3);
        }

        /* Headers and titles */
        h1, h2, h3 {
            color: #FFFFFF !important;
            font-weight: 600 !important;
            letter-spacing: -0.5px;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #161A22;
            border-right: 1px solid #2D323A;
        }
        
        /* Custom buttons */
        .stButton>button {
            border-radius: 8px;
            background: linear-gradient(90deg, #00C853 0%, #B2FF59 100%);
            color: black !important;
            font-weight: 700;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #B2FF59 0%, #00C853 100%);
            box-shadow: 0 0 15px rgba(0, 200, 83, 0.5);
            transform: scale(1.02);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            color: #A0AAB4;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(0, 200, 83, 0.1);
            color: #00C853 !important;
            border-bottom: 2px solid #00C853 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Advanced Quant Finance & Asset Pricing Engine")
st.markdown("""
*An institutional-grade portfolio analysis tool demonstrating ETL resiliency, technical indicators, and stochastic modeling.*
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

st.sidebar.markdown("---")
st.sidebar.subheader("Local Data Management")
if st.sidebar.button("Pre-download Top Stocks"):
    with st.sidebar:
        with st.spinner("Downloading AAPL, MSFT, GOOGL, TSLA, AMZN..."):
            analytics.pre_fetch_major_stocks()
        st.success("Successfully cached major stocks!")

st.sidebar.markdown("---")
st.sidebar.subheader("Custom Data (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload your own CSV data", type=['csv'])
st.sidebar.info(
    "**CSV Format Required:**\n"
    "- Your file must contain a date column named `Date`.\n"
    "- Price columns must be named: `Open`, `High`, `Low`, `Close`.\n"
    "- Optional (but recommended): `Volume`."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Technical Indicators")
sma_short = st.sidebar.slider("SMA Short Window", 5, 50, 20)
sma_long = st.sidebar.slider("SMA Long Window", 50, 200, 100)

# --- DATA LOADING ---
with st.spinner('Loading data...'):
    try:
        if uploaded_file is not None:
            # Load user provided CSV
            df = pd.read_csv(uploaded_file)
            
            # Basic validation and formatting for the uploaded file
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif not pd.api.types.is_datetime64_any_dtype(df.index):
                # Try to convert index to datetime if 'Date' column is missing
                df.index = pd.to_datetime(df.index)
                
            # Filter by selected dates
            mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
            df = df.loc[mask]
            
            if df.empty:
                st.error("No valid data found in the selected date range for this CSV.")
                st.stop()
                
            df.is_custom = True
            
        else:
            # Default behavior (Mock/Cache/Live API via analytics.py)
            df = analytics.get_stock_data(ticker, start_date, end_date)
            if df.empty:
                st.error("No data found.")
                st.stop()
        
        # Calculations
        df = analytics.calculate_rsi(df)
        df = analytics.calculate_moving_averages(df, sma_short, sma_long)
        df = analytics.calculate_bollinger_bands(df)
        
        # Prediction
        pred_df = analytics.basic_prediction(df)
        
        # Status Messages
        if getattr(df, 'is_custom', False):
            st.success("Using Custom CSV Data.")
        elif getattr(df, 'is_mock', False):
            st.warning("Using Simulated (Mock) Data. (Yahoo Finance API is currently restricted).")
        elif getattr(df, 'is_cached', False):
            st.info("Showing locally cached data (Offline mode).")
        else:
            st.success("Fresh data fetched from Live API.")
            
    except Exception as e:
        st.error(f"Error processing data: {e}. Please ensure your CSV has the correct format.")
        st.stop()

# --- METRICS ---
last_close = df['Close'].iloc[-1]
prev_close = df['Close'].iloc[-2]
change = last_close - prev_close
pct_change = (change / prev_close) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Last Price", f"${last_close:.2f}")
col2.metric("Change", f"${change:.2f}", f"{pct_change:.2f}%")
col3.metric("RSI (14d)", f"{df['RSI'].iloc[-1]:.2f}")
col4.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")

# --- MAIN CHART (Price + Bollinger + SMA) ---
st.subheader("Price Analysis")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.1, subplot_titles=('Price & Trends', 'RSI'),
                    row_width=[0.3, 0.7])

# Candlestick
fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name="Market Data"), row=1, col=1)

# SMA
fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{sma_short}'], line=dict(color='orange', width=1), name=f"SMA {sma_short}"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{sma_long}'], line=dict(color='blue', width=1), name=f"SMA {sma_long}"), row=1, col=1)

# Bollinger Bands
fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], line=dict(color='gray', width=1, dash='dash'), name="Upper Bollinger"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], line=dict(color='gray', width=1, dash='dash'), name="Lower Bollinger"), row=1, col=1)

# Linear Prediction Line
fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted_Price'], line=dict(color='red', width=2, dash='dot'), name="Trend Projection"), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig, use_container_width=True)

# --- QUANTITATIVE FINANCE SECTION ---
st.markdown("---")
st.header("Quantitative Finance Models")
st.markdown("Advanced mathematical models for risk and pricing (Monte Carlo, VaR, Black-Scholes).")

tab1, tab2, tab3 = st.tabs(["Monte Carlo Simulation", "Value at Risk (VaR)", "Options Pricing (Black-Scholes)"])

with tab1:
    st.subheader("Monte Carlo Price Path Simulation (GBM)")
    col_mc1, col_mc2 = st.columns([1, 3])
    with col_mc1:
        sim_days = st.slider("Days to Predict", 10, 252, 30)
        sim_count = st.slider("Number of Simulations", 10, 500, 100)
    
    with col_mc2:
        with st.spinner("Running simulations..."):
            mc_df = analytics.monte_carlo_simulation(df, sim_days, sim_count)
            fig_mc = go.Figure()
            for col in mc_df.columns:
                fig_mc.add_trace(go.Scatter(x=mc_df.index, y=mc_df[col], mode='lines', line=dict(width=1), opacity=0.1, showlegend=False))
            # Add historical price context
            hist_tail = df['Close'].tail(60)
            fig_mc.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail, mode='lines', line=dict(color='white', width=3), name='Historical'))
            fig_mc.update_layout(title=f"{sim_count} Simulated Paths for next {sim_days} days", template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_mc, use_container_width=True)

with tab2:
    st.subheader("Historical Value at Risk (VaR)")
    var_conf = st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)
    var_value = analytics.calculate_var(df, var_conf)
    if var_value is not None:
        st.error(f"At a {var_conf*100}% confidence level, the maximum expected daily loss is **{abs(var_value)*100:.2f}%**.")
        st.info("This means there is only a {:.0f}% probability that the asset drops more than {:.2f}% in a single day based on historical data.".format((1-var_conf)*100, abs(var_value)*100))

with tab3:
    st.subheader("Theoretical Call Option Pricing (Black-Scholes)")
    col_bs1, col_bs2 = st.columns(2)
    with col_bs1:
        current_price = df['Close'].iloc[-1].item() if not df.empty else 150.0
        S = st.number_input("Spot Price (S)", value=float(current_price))
        K = st.number_input("Strike Price (K)", value=float(current_price * 1.05))
        T = st.slider("Time to Maturity (Years)", 0.01, 2.0, 0.25)
    with col_bs2:
        r = st.number_input("Risk-Free Rate (annualized %)", value=4.5) / 100
        # Calculate historical volatility
        hist_vol = df['Close'].pct_change().std() * np.sqrt(252) if not df.empty else 0.2
        sigma = st.number_input("Implied Volatility (annualized %)", value=float(hist_vol*100)) / 100
    
    call_price = analytics.black_scholes_call(S, K, T, r, sigma)
    st.success(f"**Theoretical Call Option Price:** ${call_price:.2f}")

# --- DATA TABLE ---
st.markdown("---")
if st.checkbox("Show Raw Data"):
    st.dataframe(df.tail(20))

st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes only. Market data may be delayed. Options pricing is theoretical.")
