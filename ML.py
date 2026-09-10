import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf

# =========================================================
# 1. EDIT DASHBOARD TITLE DIRECTLY HERE IN CODE
# =========================================================
APP_TITLE = "AI Financial Analytics & Market Intelligence"

# Page Configuration (Mobile Responsive)
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="📈"
)

# Custom Styling for Pill Selector Buttons
st.markdown("""
    <style>
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        justify-content: flex-start;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 8px;
        margin-bottom: 16px;
    }
    div[role="radiogroup"] > label {
        background-color: #1a1e2a;
        padding: 6px 16px;
        border-radius: 20px;
        border: 1px solid #333846;
        color: #e0e0e0;
        cursor: pointer;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s ease;
    }
    div[role="radiogroup"] > label:hover {
        background-color: #2b3040;
        border-color: #4f5875;
    }
    div[role="radiogroup"] label[data-checked="true"] {
        background-color: #ffffff !important;
        color: #1a1e2a !important;
        border-color: #ffffff !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title & User Name Input
st.title(APP_TITLE)

col_name, _ = st.columns([1.2, 1])
with col_name:
    user_name = st.text_input("👤 Enter Your Name:", value="Investor").strip()

if not user_name:
    user_name = "Investor"

st.markdown(f"### Welcome, **{user_name}**! 👋")
st.caption("Live AI/ML stock momentum forecasts & custom Mutual Fund allocation engine.")

# Tabs
tab1, tab2 = st.tabs(["📊 Stock ML Predictor", "💼 Mutual Funds & Cap Allocation"])

# Timeframe Configuration Mapping
TIMEFRAME_CONFIG = {
    "1D": {"period": "1d", "interval": "5m", "desc": "1 Day (5-min Intraday)"},
    "1W": {"period": "5d", "interval": "15m", "desc": "1 Week (15-min)"},
    "1M": {"period": "1mo", "interval": "1h", "desc": "1 Month (Hourly)"},
    "3M": {"period": "3mo", "interval": "1d", "desc": "3 Months (Daily)"},
    "6M": {"period": "6mo", "interval": "1d", "desc": "6 Months (Daily)"},
    "1Y": {"period": "1y", "interval": "1d", "desc": "1 Year (Daily)"},
    "5Y": {"period": "5y", "interval": "1wk", "desc": "5 Years (Weekly)"},
    "All": {"period": "max", "interval": "1mo", "desc": "All Available History (Monthly)"}
}

# Cached data-fetching function (5-minute TTL to prevent API spamming)
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(symbol: str, period_val: str, interval_val: str):
    clean_sym = symbol.strip().upper()
    candidates = [clean_sym] if clean_sym.endswith(('.NS', '.BO')) else [f"{clean_sym}.NS", f"{clean_sym}.BO", clean_sym]
    
    chart_df, train_df = None, None
    resolved_ticker = clean_sym
    last_error = None
    
    for sym in candidates:
        try:
            ticker_obj = yf.Ticker(sym)
            c_df = ticker_obj.history(period=period_val, interval=interval_val)
            t_df = ticker_obj.history(period="2y", interval="1d")
            
            if c_df is not None and not c_df.empty and t_df is not None and len(t_df) >= 80:
                chart_df = c_df
                train_df = t_df
                resolved_ticker = sym
                last_error = None
                break
        except Exception as e:
            last_error = str(e)
            continue
            
    return chart_df, train_df, resolved_ticker, last_error

# =========================================================
# TAB 1: STOCK PREDICTION & MACHINE LEARNING
# =========================================================
with tab1:
    st.subheader("Real-Time Stock ML Forecast & Position Advisory")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        stock_input = st.text_input(
            "Enter Stock Symbol / Name",
            value="TCS",
            help="Examples: TCS, RELIANCE, INFY, TATAMOTORS, HDFCBANK, AAPL, TSLA"
        )
    with c2:
        forecast_days = st.selectbox("ML Prediction Horizon", [7, 14, 30], index=0)

    # --- PORTFOLIO / HOLDING INPUT SECTION ---
    st.markdown("#### 💼 Position Status")
    is_holding = st.checkbox("I already own shares of this stock")
    
    buy_price = 0.0
    shares_held = 0
    if is_holding:
        h1, h2 = st.columns(2)
        with h1:
            buy_price = st.number_input(
                "Your Average Buy Price (₹)",
                min_value=0.01,
                value=3500.0,
                step=1.0,
                help="Enter your purchase price per share"
            )
        with h2:
            shares_held = st.number_input(
                "Quantity Owned",
                min_value=1,
                value=10,
                step=1
            )

    # Timeframe Pills
    st.write("**Select Chart Timeframe:**")
    selected_tf = st.radio(
        label="Timeframe",
        options=["1D", "1W", "1M", "3M", "6M", "1Y", "5Y", "All"],
        index=2,
        horizontal=True,
        label_visibility="collapsed"
    )

    if st.button("🚀 Run Live ML Analysis", key="btn_stock", use_container_width=True):
        cfg = TIMEFRAME_CONFIG[selected_tf]
        with st.spinner(f"Fetching {cfg['desc']} data for '{stock_input}'..."):
            chart_df, train_df, resolved_ticker, fetch_err = fetch_stock_data(stock_input, cfg["period"], cfg["interval"])
            
            if chart_df is None or train_df is None:
                err_msg = f"Could not retrieve sufficient market data for '{stock_input}'."
                if fetch_err:
                    err_msg += f" (Details: {fetch_err})"
                st.error(err_msg)
            else:
                try:
                    # Clean training features
                    train_close = pd.Series(train_df['Close'].values.astype(float).flatten(), index=train_df.index)
                    
                    sma_10 = train_close.rolling(10).mean()
                    sma_50 = train_close.rolling(50).mean()
                    returns = train_close.pct_change()
                    
                    # RSI Calculation
                    delta = train_close.diff()
                    gain = delta.clip(lower=0).rolling(14).mean()
                    loss = (-delta.clip(upper=0)).rolling(14).mean()
                    rs = gain / loss.replace(0, np.nan)
                    rsi = (100 - (100 / (1 + rs)))
                    rsi = rsi.fillna(100).where(gain != 0, 50)

                    # Training DataFrame
                    ml_df = pd.DataFrame({
                        'Close': train_close,
                        'SMA_10': sma_10,
                        'SMA_50': sma_50,
                        'Return': returns,
                        'RSI': rsi
                    }).dropna()

                    ml_df['Target'] = (ml_df['Close'].shift(-forecast_days) > ml_df['Close']).astype(int)
                    
                    features = ['SMA_10', 'SMA_50', 'Return', 'RSI']
                    train_set = ml_df.iloc[:-forecast_days]
                    
                    X = train_set[features].values
                    y = train_set['Target'].values
                    
                    # Model fit & predict
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    
                    latest_row = ml_df[features].iloc[[-1]].values
                    prediction = int(model.predict(latest_row)[0])
                    probs = model.predict_proba(latest_row)[0]
                    
                    prob_dict = dict(zip(model.classes_, probs))
                    prob_up = prob_dict.get(1, 0.5)
                    prob_down = prob_dict.get(0, 0.5)

                    curr_price = float(chart_df['Close'].iloc[-1])
                    curr_rsi = float(rsi.iloc[-1])
                    curr_sma10 = float(sma_10.iloc[-1])
                    curr_sma50 = float(sma_50.iloc[-1])

                    # =========================================================
                    # HOLDING-AWARE DECISION ENGINE
                    # =========================================================
                    if is_holding:
                        pnl_per_share = curr_price - buy_price
                        total_pnl = pnl_per_share * shares_held
                        pnl_pct = (pnl_per_share / buy_price) * 100

                        if prediction == 1 and prob_up >= 0.53:
                            verdict = "HOLD"
                            color = "green"
                            reason = (
                                f"Bullish continuation expected ({prob_up*100:.1f}% confidence). "
                                f"Price is supported above SMA_10 (₹{curr_sma10:,.2f}). Continue holding your position."
                            )
                        elif prediction == 0 and prob_down >= 0.53:
                            if pnl_pct > 0:
                                verdict = "SELL / BOOK PROFIT"
                                color = "red"
                                reason = (
                                    f"Bearish momentum detected ({prob_down*100:.1f}% downside probability, RSI: {curr_rsi:.1f}). "
                                    f"Lock in your gains of {pnl_pct:.2f}% before a potential correction."
                                )
                            else:
                                verdict = "SELL / STOP-LOSS"
                                color = "red"
                                reason = (
                                    f"Downside pressure mounting ({prob_down*100:.1f}% probability, RSI: {curr_rsi:.1f}). "
                                    f"Consider cutting losses at {pnl_pct:.2f}% to preserve capital."
                                )
                        else:
                            verdict = "HOLD"
                            color = "orange"
                            reason = (
                                f"Consolidation range detected. SMA_10 (₹{curr_sma10:,.2f}) and SMA_50 (₹{curr_sma50:,.2f}) are flat. "
                                f"Maintain current position and watch for a confirmed breakout."
                            )
                    else:
                        # Non-holding standard recommendation
                        if prediction == 1 and prob_up >= 0.53:
                            verdict = "BUY"
                            color = "green"
                            reason = f"Bullish momentum detected. 10-day SMA (₹{curr_sma10:,.2f}) offers strong support with {prob_up*100:.1f}% confidence. RSI at {curr_rsi:.1f} shows expansion headroom."
                        elif prediction == 0 and prob_down >= 0.53:
                            verdict = "AVOID / SELL"
                            color = "red"
                            reason = f"Bearish divergence detected. Model signals downside probability of {prob_down*100:.1f}%. Selling pressure active (RSI: {curr_rsi:.1f})."
                        else:
                            verdict = "WAIT / HOLD"
                            color = "orange"
                            reason = f"Consolidation pattern. 10 SMA (₹{curr_sma10:,.2f}) and 50 SMA (₹{curr_sma50:,.2f}) remain range-bound without a decisive breakout trigger."

                    # Display KPI Cards
                    st.markdown("---")
                    
                    if is_holding:
                        k1, k2, k3, k4, k5 = st.columns(5)
                        k1.metric("Stock", resolved_ticker)
                        k2.metric("Market Price", f"₹{curr_price:,.2f}")
                        k3.metric("Buy Price", f"₹{buy_price:,.2f}")
                        k4.metric("Unrealized P&L", f"₹{total_pnl:,.2f}", f"{pnl_pct:.2f}%")
                        k5.metric("Model Confidence", f"{max(prob_up, prob_down)*100:.1f}%")
                    else:
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Stock Analyzed", resolved_ticker)
                        m2.metric("Latest Market Price", f"₹{curr_price:,.2f}")
                        m3.metric("ML Recommendation", verdict)
                        m4.metric("Model Confidence", f"{max(prob_up, prob_down)*100:.1f}%")

                    st.markdown(f"### **Recommendation:** :{color}[**{verdict}**]")
                    st.info(f"**ML Rationale & Position Advisory:** {reason}")

                    # Interactive Candlestick Chart
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=chart_df.index,
                        open=chart_df['Open'],
                        high=chart_df['High'],
                        low=chart_df['Low'],
                        close=chart_df['Close'],
                        name="Price"
                    ))
                    
                    chart_sma = chart_df['Close'].rolling(min(10, len(chart_df))).mean()
                    fig.add_trace(go.Scatter(
                        x=chart_df.index,
                        y=chart_sma,
                        line=dict(color='#FFA500', width=1.5),
                        name="Moving Average (10)"
                    ))

                    # Plot user buy price line if holding
                    if is_holding:
                        fig.add_hline(
                            y=buy_price,
                            line_dash="dot",
                            line_color="#00E5FF",
                            annotation_text=f"Your Buy Price (₹{buy_price:,.2f})",
                            annotation_position="bottom right"
                        )
                    
                    fig.update_layout(
                        title=f"{resolved_ticker} - Price Action ({selected_tf} View)",
                        xaxis_title="Time / Date",
                        yaxis_title="Price (₹)",
                        height=440,
                        margin=dict(l=10, r=10, t=40, b=10),
                        xaxis_rangeslider_visible=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")

# =========================================================
# TAB 2: MUTUAL FUNDS & MARKET CAP ADVISORY
# =========================================================
with tab2:
    st.subheader("Custom Mutual Fund & Market Cap Advisory")
    
    mf_col1, mf_col2, mf_col3 = st.columns(3)
    with mf_col1:
        fund_house = st.text_input("Enter Fund House / AMC Name", value="Aditya Birla Sun Life")
    with mf_col2:
        investment_amount = st.number_input("Investment Amount (₹)", min_value=500, value=25000, step=500)
    with mf_col3:
        risk_profile = st.selectbox("Your Risk Appetite", ["Conservative (Low Risk)", "Moderate (Balanced)", "Aggressive (High Growth)"])

    if st.button("Generate Fund & Cap Recommendation", key="btn_mf", use_container_width=True):
        st.markdown("---")
        
        if "Conservative" in risk_profile:
            large_cap, mid_cap, small_cap = 70, 20, 10
            cap_verdict = "Large Cap Focused"
            cap_reason = "Best suited for steady capital preservation and minimal market drawdown."
        elif "Moderate" in risk_profile:
            large_cap, mid_cap, small_cap = 40, 40, 20
            cap_verdict = "Balanced (Large & Mid Cap)"
            cap_reason = "Offers balance between large-cap stability and mid-cap compounding growth."
        else:
            large_cap, mid_cap, small_cap = 20, 40, 40
            cap_verdict = "High Growth (Mid & Small Cap)"
            cap_reason = "Aims for maximum returns over a long-term (5+ years) investment horizon."

        c_left, c_right = st.columns([1.3, 1])

        with c_left:
            st.markdown(f"#### 🎯 Recommended Portfolio for **{fund_house}**")
            fund_data = [
                {
                    "Cap Category": "Large Cap",
                    "Recommended Fund": f"{fund_house} Frontline / Large Cap Fund",
                    "Expected CAGR": "12% - 15%",
                    "Allocation (%)": f"{large_cap}%",
                    "Amount (₹)": f"₹{(investment_amount * large_cap / 100):,.0f}"
                },
                {
                    "Cap Category": "Mid Cap",
                    "Recommended Fund": f"{fund_house} Midcap Fund",
                    "Expected CAGR": "16% - 20%",
                    "Allocation (%)": f"{mid_cap}%",
                    "Amount (₹)": f"₹{(investment_amount * mid_cap / 100):,.0f}"
                },
                {
                    "Cap Category": "Small Cap",
                    "Recommended Fund": f"{fund_house} Small Cap Fund",
                    "Expected CAGR": "20% - 25%+",
                    "Allocation (%)": f"{small_cap}%",
                    "Amount (₹)": f"₹{(investment_amount * small_cap / 100):,.0f}"
                }
            ]
            st.dataframe(pd.DataFrame(fund_data), use_container_width=True, hide_index=True)

        with c_right:
            st.markdown("#### ⚖️ Portfolio Distribution")
            st.success(f"**Strategy:** {cap_verdict}")
            st.caption(cap_reason)

            pie_fig = go.Figure(data=[go.Pie(
                labels=['Large Cap', 'Mid Cap', 'Small Cap'],
                values=[large_cap, mid_cap, small_cap],
                hole=.45,
                marker=dict(colors=['#2ca02c', '#1f77b4', '#ff7f0e'])
            )])
            pie_fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(pie_fig, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 15px;'>Made with ❤️ by Parth Makwana and Ayushi Chauhan</p>", unsafe_allow_html=True)
