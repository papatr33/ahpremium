import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import statsmodels.api as sm
from scipy import stats

# --- Page Config ---
st.set_page_config(
    page_title="AH Premium",
    page_icon="📉",
    layout="wide"
)

# --- CSS ---
st.markdown("""
<style>
    .stMetricLabel {color: #666; font-size: 0.9rem; font-weight: 500;}
    .stMetricValue {color: #333; font-size: 1.6rem; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

# --- Nippon Colors Palette ---
NIPPON_COLORS = [
    '#9E3D3F', # Suoh
    '#2A5CAA', # Ruri
    '#838B0D', # Koke
    '#FFB11B', # Yamabuki
    '#5B3131', # Ebi-cha
    '#005CAF', # Rurikon
    '#C1328E', # Tsutsuji
    '#6A8372', # Byakuroku
    '#E49E61', # Araigaki
    '#4D4398', # Kon-kikyo
    '#7DB9DE', # Wasurenagusa
]

# --- Data Config ---
AH_PAIRS = {
    "Hua Hong Semiconductor": {"A": "688347.SS", "H": "1347.HK"},
    "Ping An Insurance": {"A": "601318.SS", "H": "2318.HK"},
    "CCB (China Construction Bank)": {"A": "601939.SS", "H": "0939.HK"},
    "BYD Company": {"A": "002594.SZ", "H": "1211.HK"},
    "Zijin Mining": {"A": "601899.SS", "H": "2899.HK"},
    "Ganfeng Lithium": {"A": "002460.SZ", "H": "1772.HK"},
    "China Life Insurance": {"A": "601628.SS", "H": "2628.HK"},
    "CNOOC": {"A": "600938.SS", "H": "0883.HK"},
    "China Mobile": {"A": "600941.SS", "H": "0941.HK"},
    "ICBC": {"A": "601398.SS", "H": "1398.HK"},
    "Bank of China": {"A": "601988.SS", "H": "3988.HK"},
    "ZTE": {"A": "000063.SZ", "H": "0763.HK"},
    "CM Bank (China Merchants Bank)": {"A": "600036.SS", "H": "3968.HK"},
    "YOFC (Yangtze Optical)": {"A": "601869.SS", "H": "6869.HK"},
    "Chalco (Aluminum Corp)": {"A": "601600.SS", "H": "2600.HK"},
    "PetroChina": {"A": "601857.SS", "H": "0857.HK"},
    "CMOC": {"A": "603993.SS", "H": "3993.HK"},
    "Agricultural Bank of China": {"A": "601288.SS", "H": "1288.HK"},
    "Sinopec Corp": {"A": "600028.SS", "H": "0386.HK"},
    "WuXi AppTec": {"A": "603259.SS", "H": "2359.HK"},
    "Jiangxi Copper": {"A": "600362.SS", "H": "0358.HK"},
    "NCI (New China Life)": {"A": "601336.SS", "H": "1336.HK"},
    "Tianqi Lithium": {"A": "002466.SZ", "H": "9696.HK"},
    "CPIC (China Pacific Insurance)": {"A": "601601.SS", "H": "2601.HK"},
    "China Shenhua": {"A": "601088.SS", "H": "1088.HK"},
    "RemeGen": {"A": "688331.SS", "H": "9995.HK"},
    "China Tourism Duty Free": {"A": "601888.SS", "H": "1880.HK"},
    "Lopal Tech": {"A": "603906.SS", "H": "2465.HK"},
    "Haier Smart Home": {"A": "600690.SS", "H": "6690.HK"},
    "COSCO Ship Energy": {"A": "600026.SS", "H": "1138.HK"},
    "China Telecom": {"A": "601728.SS", "H": "0728.HK"},
    "Shandong Gold": {"A": "600547.SS", "H": "1787.HK"},
    "GAC Group": {"A": "601238.SS", "H": "2238.HK"},
    "Yankuang Energy": {"A": "600188.SS", "H": "1171.HK"},
    "Weichai Power": {"A": "000338.SZ", "H": "2338.HK"},
    "COSCO Ship Holdings": {"A": "601919.SS", "H": "1919.HK"},
    "China Vanke": {"A": "000002.SZ", "H": "2202.HK"},
    "Dongfang Electric": {"A": "600875.SS", "H": "1072.HK"},
    "CITIC Securities": {"A": "600030.SS", "H": "6030.HK"},
    "PICC Group": {"A": "601319.SS", "H": "1339.HK"},
    "CATL": {"A": "300750.SZ", "H": "3750.HK"},
    "Midea": {"A": "000333.SZ", "H": "0300.HK"}
}

# --- Core Functions ---
@st.cache_data(ttl=3600)
def fetch_pair_data(a_ticker, h_ticker, start_date, end_date):
    tickers = [a_ticker, h_ticker, "CNY=X", "HKD=X"]
    try:
        raw = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='column', auto_adjust=False)
    except Exception:
        return pd.DataFrame()
    
    if raw.empty: return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        col_type = 'Adj Close' if 'Adj Close' in level0 else 'Close'
        if col_type not in level0: return pd.DataFrame()
        df = raw[col_type]
    else:
        df = raw

    try:
        temp_df = pd.DataFrame(index=df.index)
        temp_df['A_Local'] = df[a_ticker]
        temp_df['H_Local'] = df[h_ticker]
        temp_df['USDCNH'] = df['CNY=X'].ffill()
        temp_df['USDHKD'] = df['HKD=X'].ffill()
        data = temp_df.dropna(subset=['A_Local', 'H_Local'])
    except KeyError:
        return pd.DataFrame()

    return data

@st.cache_data(ttl=600)
def get_latest_spreads():
    """Sequential fetch of latest data with 1-day, 5-day, and 30-day spread changes."""
    results = []
    start_date = date.today() - timedelta(days=60)  # Extended to ensure 30-day data
    end_date = date.today() + timedelta(days=1)
    
    for name, tickers in AH_PAIRS.items():
        df = fetch_pair_data(tickers['A'], tickers['H'], start_date, end_date)
        if not df.empty and len(df) >= 2:
            df['A_USD'] = df['A_Local'] / df['USDCNH']
            df['H_USD'] = df['H_Local'] / df['USDHKD']
            df['Spread_Pct'] = ( (df['A_USD'] / df['H_USD']) - 1 ) * 100
            
            current_spread = df['Spread_Pct'].iloc[-1]
            
            # Calculate changes
            change_1d = current_spread - df['Spread_Pct'].iloc[-2] if len(df) >= 2 else np.nan
            change_5d = current_spread - df['Spread_Pct'].iloc[-6] if len(df) >= 6 else np.nan
            change_30d = current_spread - df['Spread_Pct'].iloc[-31] if len(df) >= 31 else np.nan
            
            results.append({
                "Pair": name, 
                "Current Spread (%)": current_spread,
                "1D Change (%)": change_1d,
                "5D Change (%)": change_5d,
                "30D Change (%)": change_30d
            })
    
    return pd.DataFrame(results)

def run_backtest(df, long_entry, long_exit, short_entry, short_exit, trade_size=1_000_000, enable_short=False):
    df = df.copy()
    df['A_USD'] = df['A_Local'] / df['USDCNH']
    df['H_USD'] = df['H_Local'] / df['USDHKD']
    df['Spread_Pct'] = ( (df['A_USD'] / df['H_USD']) - 1 ) * 100
    df = df.dropna()
    
    position = 0 
    cumulative_pnl = [0.0] 
    events = []
    closed_trades = []
    
    shares_a = 0.0
    shares_h = 0.0
    entry_date = None
    current_trade_pnl = 0.0
    
    for i in range(len(df)):
        row = df.iloc[i]
        today_date = df.index[i]
        spread_val = row['Spread_Pct']
        daily_pnl = 0.0
        
        if i > 0 and position != 0:
            prev_row = df.iloc[i-1]
            delta_a = row['A_USD'] - prev_row['A_USD']
            delta_h = row['H_USD'] - prev_row['H_USD']
            
            if position == 1: 
                pnl_long = delta_a * shares_a
                pnl_short = delta_h * shares_h 
                daily_pnl = pnl_long - pnl_short
            elif position == -1: 
                pnl_short = delta_a * shares_a
                pnl_long = delta_h * shares_h
                daily_pnl = pnl_long - pnl_short
            
            current_trade_pnl += daily_pnl

        cumulative_pnl.append(cumulative_pnl[-1] + daily_pnl)
            
        if position == 0:
            if spread_val < long_entry:
                position = 1
                shares_a = trade_size / row['A_USD']
                shares_h = trade_size / row['H_USD']
                entry_date = today_date
                current_trade_pnl = 0.0
                events.append({"Date": today_date, "Type": "Entry Long", "Price": spread_val, "Shares_A": shares_a, "Shares_H": shares_h})
            elif enable_short and (spread_val > short_entry):
                position = -1
                shares_a = trade_size / row['A_USD']
                shares_h = trade_size / row['H_USD']
                entry_date = today_date
                current_trade_pnl = 0.0
                events.append({"Date": today_date, "Type": "Entry Short", "Price": spread_val, "Shares_A": shares_a, "Shares_H": shares_h})
        
        elif position == 1:
            if spread_val > long_exit:
                events.append({"Date": today_date, "Type": "Exit Long", "Price": spread_val, "Shares_A": 0, "Shares_H": 0})
                closed_trades.append({"Entry Date": entry_date, "Exit Date": today_date, "Duration": (today_date - entry_date).days, "PnL": current_trade_pnl, "Type": "Long"})
                position = 0; shares_a = 0; shares_h = 0; entry_date = None
                
        elif position == -1:
            if spread_val < short_exit:
                events.append({"Date": today_date, "Type": "Exit Short", "Price": spread_val, "Shares_A": 0, "Shares_H": 0})
                closed_trades.append({"Entry Date": entry_date, "Exit Date": today_date, "Duration": (today_date - entry_date).days, "PnL": current_trade_pnl, "Type": "Short"})
                position = 0; shares_a = 0; shares_h = 0; entry_date = None
    
    df['Net_PnL'] = cumulative_pnl[1:]
    return df, pd.DataFrame(events), pd.DataFrame(closed_trades)

# --- Sidebar ---
st.sidebar.header("Strategy Config")
st.sidebar.write("Logic: Manual Thresholds")

# LONG PARAMS
st.sidebar.subheader("Long (Buy A / Sell H)")
long_entry = st.sidebar.number_input("Enter Long if Spread < (%)", value=30.0, step=1.0)
long_exit = st.sidebar.number_input("Exit Long if Spread > (%)", value=50.0, step=1.0)

st.sidebar.divider()

# SHORT PARAMS
enable_short = st.sidebar.checkbox("Enable Short Strategy?", value=False)
short_entry = st.sidebar.number_input("Enter Short if Spread > (%)", value=140.0, step=5.0, disabled=not enable_short)
short_exit = st.sidebar.number_input("Exit Short if Spread < (%)", value=120.0, step=5.0, disabled=not enable_short)

st.sidebar.divider()
st.sidebar.subheader("General Settings")
trade_size = 1_000_000
start_date_input = st.sidebar.date_input("Start Date", date(2024, 1, 1))

# --- Main App ---
st.title(f"📉 AH Premium")

tab1, tab2, tab3 = st.tabs(["Annual Stats Analysis", "Single Pair Analysis", "Rolling Correlations"])

# ==========================================
# TAB 1: Annual Stats Analysis
# ==========================================
with tab1:
    st.subheader("📊 Annual Spread Statistics & Comparison")
    
    with st.spinner("Scanning current spreads..."):
        latest_spread_df = get_latest_spreads()
    
    # --- CHART SECTION ---
    st.write("#### 1. Spread History Comparison")
    
    # User input for threshold
    spread_threshold = st.number_input(
        "Default selection threshold: Pairs with Current Spread < (%)", 
        value=10.0, 
        step=1.0,
        help="Pairs with current spread below this threshold will be auto-selected"
    )
    
    # Filter Logic: Auto-select pairs with spread < user-defined threshold
    default_selection = []
    if not latest_spread_df.empty:
        low_spread_pairs = latest_spread_df[latest_spread_df['Current Spread (%)'] < spread_threshold]
        default_selection = low_spread_pairs['Pair'].tolist()
        if not default_selection:
            default_selection = [list(AH_PAIRS.keys())[0]]
    
    selected_chart_pairs = st.multiselect(
        "Select Pairs to Compare", 
        options=list(AH_PAIRS.keys()),
        default=default_selection
    )
    
    if selected_chart_pairs:
        fig_comp = go.Figure()
        for i, p in enumerate(selected_chart_pairs):
            p_tickers = AH_PAIRS[p]
            df_p = fetch_pair_data(p_tickers['A'], p_tickers['H'], start_date_input, date.today())
            if not df_p.empty:
                df_p['A_USD'] = df_p['A_Local'] / df_p['USDCNH']
                df_p['H_USD'] = df_p['H_Local'] / df_p['USDHKD']
                spread_series = ( (df_p['A_USD'] / df_p['H_USD']) - 1 ) * 100
                color_hex = NIPPON_COLORS[i % len(NIPPON_COLORS)]
                fig_comp.add_trace(go.Scatter(x=df_p.index, y=spread_series, name=p, line=dict(color=color_hex, width=1.5)))
        
        fig_comp.update_layout(title="Historical Spread (%) Comparison", template="seaborn", hovermode="x unified", height=500)
        st.plotly_chart(fig_comp, use_container_width=True)
    
    st.divider()

    # --- CURRENT SPREAD TABLE ---
    st.write("#### 2. Current Spread Snapshot")
    if not latest_spread_df.empty:
        st.dataframe(
            latest_spread_df.sort_values(by="Current Spread (%)").style.format({
                "Current Spread (%)": "{:.2f}%",
                "1D Change (%)": "{:.2f}%",
                "5D Change (%)": "{:.2f}%",
                "30D Change (%)": "{:.2f}%"
            }).background_gradient(cmap="RdYlGn_r", subset=["Current Spread (%)"]),
            use_container_width=True, height=600, hide_index=True
        )
    else:
        st.warning("Could not fetch latest spreads.")

# ==========================================
# TAB 2: Single Pair (Visual Dashboard)
# ==========================================
with tab2:
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        selected_pair = st.selectbox("Select AH Pair", list(AH_PAIRS.keys()), key="single_pair_sel")
    pair_tickers = AH_PAIRS[selected_pair]

    with st.spinner(f"Analyzing {selected_pair}..."):
        raw_data = fetch_pair_data(pair_tickers['A'], pair_tickers['H'], start_date_input, date.today())
        
        if raw_data.empty:
            st.error("No data found.")
        else:
            res_df, event_log, closed_trades = run_backtest(
                raw_data, long_entry, long_exit, short_entry, short_exit, trade_size, enable_short
            )
            
            total_pnl = res_df['Net_PnL'].iloc[-1]
            roll_max = res_df['Net_PnL'].cummax()
            dd_dollar = (res_df['Net_PnL'] - roll_max).min()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Net Profit", f"${total_pnl:,.0f}")
            c2.metric("Max Drawdown ($)", f"${dd_dollar:,.0f}")
            c3.metric("Total Trades", len(closed_trades))
            if not closed_trades.empty:
                hit_rate = len(closed_trades[closed_trades['PnL'] > 0]) / len(closed_trades)
                c4.metric("Hit Rate", f"{hit_rate:.1%}")
            else:
                c4.metric("Hit Rate", "N/A")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
                                subplot_titles=("AH Spread & Thresholds", "Cumulative PnL ($)"))
            
            fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Spread_Pct'], name='Spread %', line=dict(color='#9E3D3F')), row=1, col=1)
            fig.add_hline(y=long_entry, line_dash="dash", line_color="green", row=1, col=1)
            fig.add_hline(y=long_exit, line_dash="dot", line_color="darkgreen", row=1, col=1)
            if enable_short:
                fig.add_hline(y=short_entry, line_dash="dash", line_color="red", row=1, col=1)
                fig.add_hline(y=short_exit, line_dash="dot", line_color="darkred", row=1, col=1)

            if not event_log.empty:
                entries = event_log[event_log['Type'].str.contains('Entry')]
                if not entries.empty:
                    fig.add_trace(go.Scatter(x=entries['Date'], y=entries['Price'], mode='markers', name='Entry', marker=dict(size=10, color='orange')), row=1, col=1)

            fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Net_PnL'], name='PnL ($)', fill='tozeroy', line=dict(color='#2A5CAA')), row=2, col=1)
            fig.update_layout(height=600, template="seaborn", margin=dict(l=40, r=40, t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            c_t1, c_t2 = st.columns(2)
            with c_t1:
                st.subheader("Recent Trades")
                if not closed_trades.empty:
                    st.dataframe(closed_trades.sort_values(by='Exit Date', ascending=False).style.format({"PnL": "${:,.0f}"}), use_container_width=True, hide_index=True)
            with c_t2:
                st.subheader("Event Log")
                if not event_log.empty:
                    st.dataframe(event_log.sort_values(by='Date', ascending=False).style.format({"Price": "{:.2f}%"}), use_container_width=True, hide_index=True)

# ==========================================
# TAB 3: Rolling Correlations (ENHANCED)
# ==========================================
with tab3:
    st.subheader("🔄 Advanced Correlation & Sensitivity Analysis")
    st.caption("Deep dive into how A-Shares, H-Shares, and the AH Spread interact over time.")
    
    col_corr_sel, col_window, col_fx = st.columns([1, 1, 1])
    with col_corr_sel:
        selected_pair_corr = st.selectbox("Select AH Pair", list(AH_PAIRS.keys()), key="corr_pair_sel")
    with col_window:
        rolling_window = st.slider("Rolling Window (days)", min_value=10, max_value=120, value=30, step=5)
    with col_fx:
        include_fx = st.toggle("Include FX in Spread", value=True, 
                               help="ON: Spread in USD terms (includes CNY/HKD movements)\nOFF: Spread in local currency terms (pure stock performance)")
    
    # Display current mode
    if include_fx:
        st.info("📊 **FX Mode: ON** — Spread calculated in USD (A_USD / H_USD). FX movements affect the spread.")
    else:
        st.warning("📊 **FX Mode: OFF** — Spread calculated in local currencies (A_CNY / H_HKD). Pure stock performance comparison.")
    
    tickers_corr = AH_PAIRS[selected_pair_corr]
    
    with st.spinner(f"Running deep analysis for {selected_pair_corr}..."):
        df_corr = fetch_pair_data(tickers_corr['A'], tickers_corr['H'], start_date_input, date.today())
        
        if not df_corr.empty and len(df_corr) > rolling_window:
            # ===========================================
            # DATA PREPARATION
            # ===========================================
            # Always calculate USD prices for reference
            df_corr['A_USD'] = df_corr['A_Local'] / df_corr['USDCNH']
            df_corr['H_USD'] = df_corr['H_Local'] / df_corr['USDHKD']
            
            # Calculate CNYHKD cross rate for local currency spread
            df_corr['CNYHKD'] = df_corr['USDHKD'] / df_corr['USDCNH']
            
            if include_fx:
                # FX-INCLUDED: Use USD-converted prices
                df_corr['Spread_Pct'] = ( (df_corr['A_USD'] / df_corr['H_USD']) - 1 ) * 100
                df_corr['Ret_A'] = df_corr['A_USD'].pct_change() * 100
                df_corr['Ret_H'] = df_corr['H_USD'].pct_change() * 100
                spread_label = "Spread (USD-based, FX included)"
            else:
                # FX-EXCLUDED: Use local currency prices
                # Spread in local terms: (A_CNY / H_HKD) * CNYHKD_rate - 1
                # This is equivalent to: (A_CNY * CNYHKD) / H_HKD - 1
                # Which gives us A in HKD terms vs H in HKD terms
                df_corr['A_HKD'] = df_corr['A_Local'] * df_corr['CNYHKD']
                df_corr['Spread_Pct'] = ( (df_corr['A_HKD'] / df_corr['H_Local']) - 1 ) * 100
                # Returns in LOCAL currency (no FX)
                df_corr['Ret_A'] = df_corr['A_Local'].pct_change() * 100  # CNY returns
                df_corr['Ret_H'] = df_corr['H_Local'].pct_change() * 100  # HKD returns
                spread_label = "Spread (Local currency, FX excluded)"
            
            df_corr['Ret_Spread'] = df_corr['Spread_Pct'].diff()  # Spread change in points
            
            # Relative return (A outperformance)
            df_corr['Ret_Diff'] = df_corr['Ret_A'] - df_corr['Ret_H']
            
            # Realized volatility
            df_corr['Vol_A'] = df_corr['Ret_A'].rolling(rolling_window).std()
            df_corr['Vol_H'] = df_corr['Ret_H'].rolling(rolling_window).std()
            df_corr['Vol_Spread'] = df_corr['Ret_Spread'].rolling(rolling_window).std()
            
            # Clean data for regression
            df_clean = df_corr.dropna(subset=['Ret_A', 'Ret_H', 'Ret_Spread'])
            
            # ===========================================
            # SECTION 1: SUMMARY STATISTICS PANEL
            # ===========================================
            st.markdown("---")
            fx_status = "FX Included" if include_fx else "FX Excluded"
            st.markdown(f"### 📊 Summary Statistics (Full Period) — {fx_status}")
            
            # Full period correlations
            corr_ah = df_clean['Ret_A'].corr(df_clean['Ret_H'])
            corr_spread_a = df_clean['Ret_Spread'].corr(df_clean['Ret_A'])
            corr_spread_h = df_clean['Ret_Spread'].corr(df_clean['Ret_H'])
            
            # Beta regression: Spread Change = α + β_A × Ret_A + β_H × Ret_H
            X_reg = df_clean[['Ret_A', 'Ret_H']]
            X_reg = sm.add_constant(X_reg)
            y_reg = df_clean['Ret_Spread']
            
            try:
                model = sm.OLS(y_reg, X_reg).fit()
                beta_a = model.params.get('Ret_A', np.nan)
                beta_h = model.params.get('Ret_H', np.nan)
                r_squared = model.rsquared
                pval_a = model.pvalues.get('Ret_A', np.nan)
                pval_h = model.pvalues.get('Ret_H', np.nan)
            except:
                beta_a, beta_h, r_squared, pval_a, pval_h = np.nan, np.nan, np.nan, np.nan, np.nan
            
            # Display metrics
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Corr(A, H)", f"{corr_ah:.3f}", help="Correlation between A and H share returns")
            with col_s2:
                st.metric("Corr(Spread, A)", f"{corr_spread_a:.3f}", help="Correlation between spread change and A returns")
            with col_s3:
                st.metric("Corr(Spread, H)", f"{corr_spread_h:.3f}", help="Correlation between spread change and H returns")
            with col_s4:
                st.metric("Model R²", f"{r_squared:.3f}", help="How much of spread movement is explained by A & H returns")
            
            # Beta display
            st.markdown("#### 🎯 Sensitivity Analysis (Beta Coefficients)")
            st.caption("Regression: `Spread Change = α + β_A × A_Return + β_H × H_Return`")
            
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                sig_a = "✓" if pval_a < 0.05 else "✗"
                st.metric(f"β_A (A-Share Beta) {sig_a}", f"{beta_a:.4f}", 
                         help=f"For 1% A-share move, spread changes by {beta_a:.2f} points. p-value: {pval_a:.4f}")
            with col_b2:
                sig_h = "✓" if pval_h < 0.05 else "✗"
                st.metric(f"β_H (H-Share Beta) {sig_h}", f"{beta_h:.4f}",
                         help=f"For 1% H-share move, spread changes by {beta_h:.2f} points. p-value: {pval_h:.4f}")
            with col_b3:
                # Relative contribution
                abs_beta_a = abs(beta_a) if not np.isnan(beta_a) else 0
                abs_beta_h = abs(beta_h) if not np.isnan(beta_h) else 0
                total_beta = abs_beta_a + abs_beta_h
                contrib_a = (abs_beta_a / total_beta * 100) if total_beta > 0 else 50
                st.metric("A-Share Contribution", f"{contrib_a:.1f}%",
                         help="Relative weight of A-share in driving spread changes")
            
            # Interpretation
            with st.expander("📖 How to interpret these betas"):
                if include_fx:
                    st.markdown("""
                    **FX Mode: ON (USD-based spread)**
                    
                    **Expected Values (by definition of spread = A_USD/H_USD - 1):**
                    - **β_A should be positive (~1.0)**: When A rises in USD, spread widens
                    - **β_H should be negative (~-1.0)**: When H rises in USD, spread narrows
                    
                    **What deviations mean:**
                    - If |β_A| > |β_H|: A-share movements dominate spread dynamics
                    - If |β_H| > |β_A|: H-share movements dominate spread dynamics  
                    - If R² is low: Other factors (sentiment, flows) drive the spread
                    
                    **Note:** In this mode, FX movements are embedded in the returns. CNY appreciation will show up as positive A-share "return" even if local price is flat.
                    """)
                else:
                    st.markdown("""
                    **FX Mode: OFF (Local currency spread)**
                    
                    **What you're measuring:**
                    - A-share returns in **CNY** (local price changes only)
                    - H-share returns in **HKD** (local price changes only)
                    - Spread is the ratio adjusted for CNYHKD cross-rate
                    
                    **This isolates pure stock performance:**
                    - β_A and β_H now measure sensitivity to LOCAL stock moves
                    - FX is completely removed from the analysis
                    - Useful for understanding fundamental stock dynamics
                    
                    **Compare with FX ON to see:**
                    - How much of spread movement comes from FX vs. stock performance
                    - Whether your edge is in stock picking or FX views
                    """)
            
            # ===========================================
            # SECTION 2: ROLLING CORRELATIONS
            # ===========================================
            st.markdown("---")
            st.markdown(f"### 📈 Rolling Correlations ({rolling_window}-Day Window) — {fx_status}")
            
            # Rolling correlations
            df_corr['Roll_Corr_AH'] = df_corr['Ret_A'].rolling(rolling_window).corr(df_corr['Ret_H'])
            df_corr['Roll_Corr_SprA'] = df_corr['Ret_Spread'].rolling(rolling_window).corr(df_corr['Ret_A'])
            df_corr['Roll_Corr_SprH'] = df_corr['Ret_Spread'].rolling(rolling_window).corr(df_corr['Ret_H'])
            
            # Chart 1: A vs H Returns Correlation
            fig_ah = go.Figure()
            fig_ah.add_trace(go.Scatter(x=df_corr.index, y=df_corr['Roll_Corr_AH'], 
                                        name=f'{rolling_window}-Day Corr', line=dict(color='#2A5CAA', width=2)))
            fig_ah.add_hline(y=corr_ah, line_dash="dash", line_color="gray", 
                            annotation_text=f"Avg: {corr_ah:.2f}")
            fig_ah.update_layout(
                title="A-Share vs H-Share Returns Correlation",
                yaxis_title="Correlation", yaxis_range=[-1, 1],
                template="seaborn", height=350, hovermode="x unified"
            )
            st.plotly_chart(fig_ah, use_container_width=True)
            
            c_c1, c_c2 = st.columns(2)
            with c_c1:
                fig_sa = go.Figure()
                fig_sa.add_trace(go.Scatter(x=df_corr.index, y=df_corr['Roll_Corr_SprA'], 
                                           name='Corr(Spread, A)', line=dict(color='#838B0D', width=1.5)))
                fig_sa.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_sa.update_layout(title="Spread Change vs A-Share Returns", 
                                    yaxis_range=[-1, 1], template="seaborn", height=300)
                st.plotly_chart(fig_sa, use_container_width=True)
            
            with c_c2:
                fig_sh = go.Figure()
                fig_sh.add_trace(go.Scatter(x=df_corr.index, y=df_corr['Roll_Corr_SprH'], 
                                           name='Corr(Spread, H)', line=dict(color='#C1328E', width=1.5)))
                fig_sh.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_sh.update_layout(title="Spread Change vs H-Share Returns", 
                                    yaxis_range=[-1, 1], template="seaborn", height=300)
                st.plotly_chart(fig_sh, use_container_width=True)
            
            # ===========================================
            # SECTION 3: LEAD-LAG ANALYSIS
            # ===========================================
            st.markdown("---")
            st.markdown("### ⏱️ Lead-Lag Analysis")
            st.caption("Does one market predict the other? Positive lag = A leads H (or vice versa)")
            
            max_lag = 10
            lags = list(range(-max_lag, max_lag + 1))
            
            # Cross-correlations
            cross_corr_ah = []  # Does A predict H?
            cross_corr_ha = []  # Does H predict A?
            cross_corr_spread_a = []  # Does spread predict A returns?
            cross_corr_a_spread = []  # Does A return predict spread?
            
            for lag in lags:
                if lag == 0:
                    cross_corr_ah.append(df_clean['Ret_A'].corr(df_clean['Ret_H']))
                    cross_corr_spread_a.append(df_clean['Ret_Spread'].corr(df_clean['Ret_A']))
                    cross_corr_a_spread.append(df_clean['Ret_A'].corr(df_clean['Ret_Spread']))
                elif lag > 0:
                    # Positive lag: past A predicts future H
                    cross_corr_ah.append(df_clean['Ret_A'].shift(lag).corr(df_clean['Ret_H']))
                    cross_corr_spread_a.append(df_clean['Ret_Spread'].shift(lag).corr(df_clean['Ret_A']))
                    cross_corr_a_spread.append(df_clean['Ret_A'].shift(lag).corr(df_clean['Ret_Spread']))
                else:
                    # Negative lag: future A correlates with past H (H leads A)
                    cross_corr_ah.append(df_clean['Ret_A'].shift(lag).corr(df_clean['Ret_H']))
                    cross_corr_spread_a.append(df_clean['Ret_Spread'].shift(lag).corr(df_clean['Ret_A']))
                    cross_corr_a_spread.append(df_clean['Ret_A'].shift(lag).corr(df_clean['Ret_Spread']))
            
            col_lag1, col_lag2 = st.columns(2)
            
            with col_lag1:
                fig_lag1 = go.Figure()
                fig_lag1.add_trace(go.Bar(x=lags, y=cross_corr_ah, marker_color='#2A5CAA', name='Cross-Corr'))
                fig_lag1.add_hline(y=0, line_color="gray")
                fig_lag1.update_layout(
                    title="A-Share → H-Share Lead-Lag",
                    xaxis_title="Lag (days, positive = A leads)",
                    yaxis_title="Correlation",
                    template="seaborn", height=300
                )
                st.plotly_chart(fig_lag1, use_container_width=True)
                
                # Find strongest lag
                max_idx = np.argmax(np.abs(cross_corr_ah))
                best_lag = lags[max_idx]
                best_corr = cross_corr_ah[max_idx]
                if best_lag > 0:
                    st.info(f"📌 Strongest signal at lag {best_lag}: A-share leads H-share by {best_lag} day(s) (corr: {best_corr:.3f})")
                elif best_lag < 0:
                    st.info(f"📌 Strongest signal at lag {best_lag}: H-share leads A-share by {abs(best_lag)} day(s) (corr: {best_corr:.3f})")
                else:
                    st.info(f"📌 Strongest signal is contemporaneous (lag 0, corr: {best_corr:.3f})")
            
            with col_lag2:
                fig_lag2 = go.Figure()
                fig_lag2.add_trace(go.Bar(x=lags, y=cross_corr_a_spread, marker_color='#9E3D3F', name='Cross-Corr'))
                fig_lag2.add_hline(y=0, line_color="gray")
                fig_lag2.update_layout(
                    title="A-Share Returns → Spread Change Lead-Lag",
                    xaxis_title="Lag (days, positive = A leads spread)",
                    yaxis_title="Correlation",
                    template="seaborn", height=300
                )
                st.plotly_chart(fig_lag2, use_container_width=True)
                
                # Interpretation
                max_idx2 = np.argmax(np.abs(cross_corr_a_spread))
                best_lag2 = lags[max_idx2]
                best_corr2 = cross_corr_a_spread[max_idx2]
                if best_lag2 > 0:
                    st.info(f"📌 A-share returns predict spread changes {best_lag2} day(s) later (corr: {best_corr2:.3f})")
                elif best_lag2 < 0:
                    st.info(f"📌 Spread changes predict A-share returns {abs(best_lag2)} day(s) later (corr: {best_corr2:.3f})")
                else:
                    st.info(f"📌 A-share and spread move together (contemporaneous, corr: {best_corr2:.3f})")
            
            # ===========================================
            # SECTION 4: SCATTER PLOTS & DISTRIBUTION
            # ===========================================
            st.markdown("---")
            st.markdown("### 🔬 Relationship Visualization")
            
            col_sc1, col_sc2 = st.columns(2)
            
            with col_sc1:
                # Scatter: A returns vs Spread change
                fig_scatter1 = go.Figure()
                fig_scatter1.add_trace(go.Scatter(
                    x=df_clean['Ret_A'], y=df_clean['Ret_Spread'],
                    mode='markers', marker=dict(color='#2A5CAA', size=5, opacity=0.5),
                    name='Daily observations'
                ))
                # Add regression line
                slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(df_clean['Ret_A'], df_clean['Ret_Spread'])
                x_line = np.linspace(df_clean['Ret_A'].min(), df_clean['Ret_A'].max(), 100)
                fig_scatter1.add_trace(go.Scatter(
                    x=x_line, y=slope_a * x_line + intercept_a,
                    mode='lines', line=dict(color='red', width=2),
                    name=f'β={slope_a:.3f}'
                ))
                fig_scatter1.update_layout(
                    title="A-Share Returns vs Spread Change",
                    xaxis_title="A-Share Return (%)", yaxis_title="Spread Change (pts)",
                    template="seaborn", height=350
                )
                st.plotly_chart(fig_scatter1, use_container_width=True)
            
            with col_sc2:
                # Scatter: H returns vs Spread change
                fig_scatter2 = go.Figure()
                fig_scatter2.add_trace(go.Scatter(
                    x=df_clean['Ret_H'], y=df_clean['Ret_Spread'],
                    mode='markers', marker=dict(color='#C1328E', size=5, opacity=0.5),
                    name='Daily observations'
                ))
                slope_h, intercept_h, r_h, p_h, se_h = stats.linregress(df_clean['Ret_H'], df_clean['Ret_Spread'])
                x_line_h = np.linspace(df_clean['Ret_H'].min(), df_clean['Ret_H'].max(), 100)
                fig_scatter2.add_trace(go.Scatter(
                    x=x_line_h, y=slope_h * x_line_h + intercept_h,
                    mode='lines', line=dict(color='red', width=2),
                    name=f'β={slope_h:.3f}'
                ))
                fig_scatter2.update_layout(
                    title="H-Share Returns vs Spread Change",
                    xaxis_title="H-Share Return (%)", yaxis_title="Spread Change (pts)",
                    template="seaborn", height=350
                )
                st.plotly_chart(fig_scatter2, use_container_width=True)
            
            # ===========================================
            # SECTION 5: REGIME ANALYSIS
            # ===========================================
            st.markdown("---")
            st.markdown("### 🌡️ Regime Analysis: Volatility Impact")
            st.caption("Do correlations change in high vs low volatility environments?")
            
            # Calculate rolling volatility for regime classification
            df_regime = df_corr.dropna(subset=['Vol_A', 'Vol_H', 'Ret_A', 'Ret_H', 'Ret_Spread'])
            
            if len(df_regime) > 50:
                # Define regimes based on combined volatility
                df_regime['Combined_Vol'] = (df_regime['Vol_A'] + df_regime['Vol_H']) / 2
                vol_median = df_regime['Combined_Vol'].median()
                
                df_high_vol = df_regime[df_regime['Combined_Vol'] > vol_median]
                df_low_vol = df_regime[df_regime['Combined_Vol'] <= vol_median]
                
                # Calculate correlations for each regime
                metrics_high = {
                    'Corr(A,H)': df_high_vol['Ret_A'].corr(df_high_vol['Ret_H']),
                    'Corr(Spread,A)': df_high_vol['Ret_Spread'].corr(df_high_vol['Ret_A']),
                    'Corr(Spread,H)': df_high_vol['Ret_Spread'].corr(df_high_vol['Ret_H']),
                    'Observations': len(df_high_vol)
                }
                
                metrics_low = {
                    'Corr(A,H)': df_low_vol['Ret_A'].corr(df_low_vol['Ret_H']),
                    'Corr(Spread,A)': df_low_vol['Ret_Spread'].corr(df_low_vol['Ret_A']),
                    'Corr(Spread,H)': df_low_vol['Ret_Spread'].corr(df_low_vol['Ret_H']),
                    'Observations': len(df_low_vol)
                }
                
                # Display as comparison table
                regime_df = pd.DataFrame({
                    'Metric': ['Corr(A, H)', 'Corr(Spread, A)', 'Corr(Spread, H)', 'Observations'],
                    'High Volatility': [metrics_high['Corr(A,H)'], metrics_high['Corr(Spread,A)'], 
                                       metrics_high['Corr(Spread,H)'], metrics_high['Observations']],
                    'Low Volatility': [metrics_low['Corr(A,H)'], metrics_low['Corr(Spread,A)'],
                                      metrics_low['Corr(Spread,H)'], metrics_low['Observations']]
                })
                
                col_reg1, col_reg2 = st.columns([1, 1])
                
                with col_reg1:
                    st.dataframe(regime_df.style.format({
                        'High Volatility': lambda x: f'{x:.3f}' if isinstance(x, float) else f'{x:,}',
                        'Low Volatility': lambda x: f'{x:.3f}' if isinstance(x, float) else f'{x:,}'
                    }), use_container_width=True, hide_index=True)
                
                with col_reg2:
                    # Bar chart comparison
                    fig_regime = go.Figure()
                    metrics_names = ['Corr(A,H)', 'Corr(Spread,A)', 'Corr(Spread,H)']
                    high_vals = [metrics_high['Corr(A,H)'], metrics_high['Corr(Spread,A)'], metrics_high['Corr(Spread,H)']]
                    low_vals = [metrics_low['Corr(A,H)'], metrics_low['Corr(Spread,A)'], metrics_low['Corr(Spread,H)']]
                    
                    fig_regime.add_trace(go.Bar(name='High Vol', x=metrics_names, y=high_vals, marker_color='#9E3D3F'))
                    fig_regime.add_trace(go.Bar(name='Low Vol', x=metrics_names, y=low_vals, marker_color='#2A5CAA'))
                    fig_regime.update_layout(barmode='group', template="seaborn", height=300,
                                            title="Correlation by Volatility Regime")
                    st.plotly_chart(fig_regime, use_container_width=True)
                
                # Interpretation
                diff_ah = metrics_high['Corr(A,H)'] - metrics_low['Corr(A,H)']
                if diff_ah > 0.1:
                    st.success("📊 A-H correlation **increases** in high volatility periods (markets move together more during stress)")
                elif diff_ah < -0.1:
                    st.warning("📊 A-H correlation **decreases** in high volatility periods (markets diverge during stress)")
                else:
                    st.info("📊 A-H correlation is **stable** across volatility regimes")
            
            # ===========================================
            # SECTION 6: ROLLING BETAS
            # ===========================================
            st.markdown("---")
            st.markdown(f"### 📉 Rolling Beta Analysis ({rolling_window}-Day Window)")
            st.caption("How the sensitivity of spread to each share class changes over time")
            
            # Calculate rolling betas using rolling regression
            rolling_beta_a = []
            rolling_beta_h = []
            rolling_r2 = []
            roll_dates = []
            
            for i in range(rolling_window, len(df_clean)):
                window_data = df_clean.iloc[i-rolling_window:i]
                try:
                    X_roll = window_data[['Ret_A', 'Ret_H']]
                    X_roll = sm.add_constant(X_roll)
                    y_roll = window_data['Ret_Spread']
                    model_roll = sm.OLS(y_roll, X_roll).fit()
                    rolling_beta_a.append(model_roll.params.get('Ret_A', np.nan))
                    rolling_beta_h.append(model_roll.params.get('Ret_H', np.nan))
                    rolling_r2.append(model_roll.rsquared)
                    roll_dates.append(df_clean.index[i])
                except:
                    rolling_beta_a.append(np.nan)
                    rolling_beta_h.append(np.nan)
                    rolling_r2.append(np.nan)
                    roll_dates.append(df_clean.index[i])
            
            fig_roll_beta = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                          subplot_titles=("Rolling Betas (β_A and β_H)", "Rolling R²"))
            
            fig_roll_beta.add_trace(go.Scatter(x=roll_dates, y=rolling_beta_a, name='β_A (A-Share)', 
                                               line=dict(color='#2A5CAA', width=1.5)), row=1, col=1)
            fig_roll_beta.add_trace(go.Scatter(x=roll_dates, y=rolling_beta_h, name='β_H (H-Share)', 
                                               line=dict(color='#9E3D3F', width=1.5)), row=1, col=1)
            fig_roll_beta.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
            
            fig_roll_beta.add_trace(go.Scatter(x=roll_dates, y=rolling_r2, name='R²', 
                                               fill='tozeroy', line=dict(color='#838B0D', width=1)), row=2, col=1)
            
            fig_roll_beta.update_layout(height=500, template="seaborn", hovermode="x unified")
            fig_roll_beta.update_yaxes(title_text="Beta", row=1, col=1)
            fig_roll_beta.update_yaxes(title_text="R²", range=[0, 1], row=2, col=1)
            st.plotly_chart(fig_roll_beta, use_container_width=True)
            
            with st.expander("📖 How to use Rolling Betas for trading"):
                st.markdown("""
                **Stable betas (low variance over time):**
                - Spread behavior is predictable
                - Mean-reversion strategies more reliable
                
                **Unstable betas (high variance):**
                - Market microstructure is changing
                - Be cautious with spread trades
                
                **High R² periods:**
                - Spread is fully explained by A & H performance
                - Pure relative value play
                
                **Low R² periods:**
                - External factors (flows, sentiment, FX) are moving the spread
                - Potential alpha opportunities if you understand the drivers
                """)
            
        else:
            st.error("Insufficient data for correlation analysis. Need at least {rolling_window} days.")

