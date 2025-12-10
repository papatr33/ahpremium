import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

# --- Page Config ---
st.set_page_config(
    page_title="AH Premium: Strategy Backtest",
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
}

# --- Core Functions ---
@st.cache_data(ttl=3600)
def fetch_pair_data(a_ticker, h_ticker, start_date, end_date):
    tickers = [a_ticker, h_ticker, "CNY=X", "HKD=X"]
    try:
        raw = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='column')
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

def run_backtest(df, long_entry, long_exit, short_entry, short_exit, trade_size=1_000_000, enable_short=False):
    """
    Backtest with Manual Thresholds.
    """
    df = df.copy()
    df['A_USD'] = df['A_Local'] / df['USDCNH']
    df['H_USD'] = df['H_Local'] / df['USDHKD']
    df['Spread_Pct'] = ( (df['A_USD'] / df['H_USD']) - 1 ) * 100
    df = df.dropna()
    
    position = 0 
    cumulative_pnl = [0.0] 
    
    # Tracking for Trade Stats
    events = []
    closed_trades = []
    
    # Trade State
    shares_a = 0.0
    shares_h = 0.0
    entry_date = None
    current_trade_pnl = 0.0
    
    for i in range(len(df)):
        row = df.iloc[i]
        today_date = df.index[i]
        spread_val = row['Spread_Pct']
        
        daily_pnl = 0.0
        
        # 1. Mark-to-Market PnL
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
            
        # 2. Trade Logic
        if position == 0:
            # Check LONG Entry
            if spread_val < long_entry:
                position = 1
                shares_a = trade_size / row['A_USD']
                shares_h = trade_size / row['H_USD']
                entry_date = today_date
                current_trade_pnl = 0.0 # Reset for new trade
                
                events.append({
                    "Date": today_date, "Type": "Entry Long", "Price": spread_val, 
                    "Shares_A": shares_a, "Shares_H": shares_h
                })
            
            # Check SHORT Entry
            elif enable_short and (spread_val > short_entry):
                position = -1
                shares_a = trade_size / row['A_USD']
                shares_h = trade_size / row['H_USD']
                entry_date = today_date
                current_trade_pnl = 0.0
                
                events.append({
                    "Date": today_date, "Type": "Entry Short", "Price": spread_val, 
                    "Shares_A": shares_a, "Shares_H": shares_h
                })
        
        elif position == 1:
            # Check LONG Exit
            if spread_val > long_exit:
                events.append({
                    "Date": today_date, "Type": "Exit Long", "Price": spread_val, 
                    "Shares_A": 0, "Shares_H": 0
                })
                # Log Closed Trade
                closed_trades.append({
                    "Entry Date": entry_date,
                    "Exit Date": today_date,
                    "Duration": (today_date - entry_date).days,
                    "PnL": current_trade_pnl,
                    "Type": "Long"
                })
                position = 0; shares_a = 0; shares_h = 0; entry_date = None
                
        elif position == -1:
            # Check SHORT Exit
            if spread_val < short_exit:
                events.append({
                    "Date": today_date, "Type": "Exit Short", "Price": spread_val, 
                    "Shares_A": 0, "Shares_H": 0
                })
                # Log Closed Trade
                closed_trades.append({
                    "Entry Date": entry_date,
                    "Exit Date": today_date,
                    "Duration": (today_date - entry_date).days,
                    "PnL": current_trade_pnl,
                    "Type": "Short"
                })
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
start_date_input = st.sidebar.date_input("Start Date", date(2016, 1, 1))

# --- Main App ---
st.title(f"📉 AH Premium: Fixed Threshold Backtest")

tab1, tab2, tab3 = st.tabs(["Single Pair Analysis", "Batch Strategy Summary", "Annual Stats Analysis"])

# ==========================================
# TAB 1: Single Pair (Visual Dashboard)
# ==========================================
with tab1:
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        selected_pair = st.selectbox("Select AH Pair", list(AH_PAIRS.keys()))
    pair_tickers = AH_PAIRS[selected_pair]

    with st.spinner(f"Analyzing {selected_pair}..."):
        raw_data = fetch_pair_data(pair_tickers['A'], pair_tickers['H'], start_date_input, date.today())
        
        if raw_data.empty:
            st.error("No data found.")
        else:
            res_df, event_log, closed_trades = run_backtest(
                raw_data, 
                long_entry, long_exit, 
                short_entry, short_exit, 
                trade_size, enable_short
            )
            
            # Metrics
            total_pnl = res_df['Net_PnL'].iloc[-1]
            roll_max = res_df['Net_PnL'].cummax()
            dd_dollar = (res_df['Net_PnL'] - roll_max).min()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Net Profit", f"${total_pnl:,.0f}")
            c2.metric("Max Drawdown ($)", f"${dd_dollar:,.0f}", help="Maximum Peak-to-Trough Dollar Loss")
            c3.metric("Total Trades", len(closed_trades))
            
            if not closed_trades.empty:
                hit_rate = len(closed_trades[closed_trades['PnL'] > 0]) / len(closed_trades)
                c4.metric("Hit Rate", f"{hit_rate:.1%}")
            else:
                c4.metric("Hit Rate", "N/A")

            # --- Plots ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
                                subplot_titles=("AH Spread & Thresholds", "Cumulative PnL ($)"))
            
            # 1. Spread Line
            fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Spread_Pct'], name='Spread %', line=dict(color='#4c72b0')), row=1, col=1)
            
            # 2. Threshold Lines
            fig.add_hline(y=long_entry, line_dash="dash", line_color="green", annotation_text="Long Entry", annotation_position="top right", row=1, col=1)
            fig.add_hline(y=long_exit, line_dash="dot", line_color="darkgreen", annotation_text="Long Exit", annotation_position="bottom right", row=1, col=1)
            
            if enable_short:
                fig.add_hline(y=short_entry, line_dash="dash", line_color="red", annotation_text="Short Entry", annotation_position="bottom right", row=1, col=1)
                fig.add_hline(y=short_exit, line_dash="dot", line_color="darkred", annotation_text="Short Exit", annotation_position="top right", row=1, col=1)

            # 3. Trade Markers
            if not event_log.empty:
                entries_long = event_log[event_log['Type'] == 'Entry Long']
                if not entries_long.empty:
                    fig.add_trace(go.Scatter(x=entries_long['Date'], y=entries_long['Price'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=12, color='green')), row=1, col=1)
                
                entries_short = event_log[event_log['Type'] == 'Entry Short']
                if not entries_short.empty:
                    fig.add_trace(go.Scatter(x=entries_short['Date'], y=entries_short['Price'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=12, color='red')), row=1, col=1)
                
                exits = event_log[event_log['Type'].str.contains('Exit')]
                if not exits.empty:
                    fig.add_trace(go.Scatter(x=exits['Date'], y=exits['Price'], mode='markers', name='Exit', marker=dict(symbol='x', size=10, color='black')), row=1, col=1)

            # 4. PnL Line
            fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Net_PnL'], name='PnL ($)', fill='tozeroy', line=dict(color='#6495ed')), row=2, col=1)
            
            fig.update_layout(height=600, template="seaborn", margin=dict(l=40, r=40, t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # --- Trade List ---
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.subheader("Recent Closed Trades")
                if not closed_trades.empty:
                    st.dataframe(closed_trades.sort_values(by='Exit Date', ascending=False).style.format({"PnL": "${:,.0f}"}), use_container_width=True, hide_index=True)
                else:
                    st.info("No closed trades.")
            
            with col_t2:
                st.subheader("Event Log")
                if not event_log.empty:
                    st.dataframe(event_log.sort_values(by='Date', ascending=False).style.format({"Price": "{:.2f}%"}), use_container_width=True, hide_index=True)


# ==========================================
# TAB 2: Batch Summary
# ==========================================
with tab2:
    st.subheader("🔎 Batch Strategy Summary")
    st.markdown("Run the current strategy settings against **all** AH pairs.")
    
    if st.button("Run Batch Backtest"):
        summary_data = []
        progress_bar = st.progress(0)
        pairs_items = list(AH_PAIRS.items())
        
        for idx, (name, tickers) in enumerate(pairs_items):
            df_curr = fetch_pair_data(tickers['A'], tickers['H'], start_date_input, date.today())
            
            if not df_curr.empty:
                res, _, trades = run_backtest(
                    df_curr, long_entry, long_exit, short_entry, short_exit, trade_size, enable_short
                )
                
                total_trades = len(trades)
                total_pnl = res['Net_PnL'].iloc[-1]
                
                roll_max = res['Net_PnL'].cummax()
                max_dd_dollar = (res['Net_PnL'] - roll_max).min()
                
                longest_horizon = trades['Duration'].max() if not trades.empty else 0
                hit_rate = (len(trades[trades['PnL'] > 0]) / total_trades) if total_trades > 0 else 0
                
                summary_data.append({
                    "Pair": name,
                    "Total Trades": total_trades,
                    "Longest Horizon (Days)": longest_horizon,
                    "Total PnL ($)": total_pnl,
                    "Max DD ($)": max_dd_dollar,
                    "Hit Rate": hit_rate
                })
            
            progress_bar.progress((idx + 1) / len(pairs_items))
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            
            st.dataframe(
                df_summary.style
                .format({
                    "Total PnL ($)": "${:,.0f}",
                    "Max DD ($)": "${:,.0f}",
                    "Hit Rate": "{:.1%}",
                    "Longest Horizon (Days)": "{:.0f}"
                }, na_rep="")
                .background_gradient(subset=["Total PnL ($)", "Hit Rate"], cmap="RdYlGn")
                .background_gradient(subset=["Max DD ($)"], cmap="Reds_r")
                .highlight_null(props="background-color: white; color: black;"), 
                use_container_width=True,
                height=800
            )

# ==========================================
# TAB 3: Annual Stats Analysis (Updated)
# ==========================================
with tab3:
    st.subheader("📊 Annual Spread Statistics & Comparison")
    
    # --- 1. Multi-Pair Chart Section ---
    st.write("#### 1. Spread History Comparison")
    
    selected_chart_pairs = st.multiselect(
        "Select Pairs to Compare (Max 10 recommended)", 
        options=list(AH_PAIRS.keys()),
        default=[list(AH_PAIRS.keys())[0], list(AH_PAIRS.keys())[1]]
    )
    
    if selected_chart_pairs:
        fig_comp = go.Figure()
        
        for p in selected_chart_pairs:
            p_tickers = AH_PAIRS[p]
            df_p = fetch_pair_data(p_tickers['A'], p_tickers['H'], start_date_input, date.today())
            if not df_p.empty:
                 # Calculate Spread
                df_p['A_USD'] = df_p['A_Local'] / df_p['USDCNH']
                df_p['H_USD'] = df_p['H_Local'] / df_p['USDHKD']
                spread_series = ( (df_p['A_USD'] / df_p['H_USD']) - 1 ) * 100
                fig_comp.add_trace(go.Scatter(x=df_p.index, y=spread_series, name=p))
        
        fig_comp.update_layout(title="Historical Spread (%) Comparison", template="seaborn", hovermode="x unified", height=500)
        st.plotly_chart(fig_comp, use_container_width=True)
    
    st.divider()

    # --- 2. Heatmap Section ---
    st.write("#### 2. Annual Heatmap Generator")
    if st.button("Generate Annual Stats Matrix"):
        all_stats = []
        progress_bar = st.progress(0)
        pairs_items = list(AH_PAIRS.items())
        
        for idx, (name, tickers) in enumerate(pairs_items):
            df_curr = fetch_pair_data(tickers['A'], tickers['H'], start_date_input, date.today())
            
            if not df_curr.empty:
                df_curr['A_USD'] = df_curr['A_Local'] / df_curr['USDCNH']
                df_curr['H_USD'] = df_curr['H_Local'] / df_curr['USDHKD']
                df_curr['Spread_Pct'] = ( (df_curr['A_USD'] / df_curr['H_USD']) - 1 ) * 100
                df_curr['Year'] = df_curr.index.year
                
                grouped = df_curr.groupby('Year')['Spread_Pct']
                
                annual_means = grouped.mean()
                annual_stds = grouped.std()
                annual_mins = grouped.min()
                annual_maxs = grouped.max()
                
                # New: Calculate Difference
                annual_diffs = annual_maxs - annual_mins
                
                unique_years = df_curr['Year'].unique()
                for y in unique_years:
                    all_stats.append({
                        "Pair": name,
                        "Year": y,
                        "Average Spread": annual_means.get(y, np.nan),
                        "Spread Volatility": annual_stds.get(y, np.nan),
                        "Min Spread": annual_mins.get(y, np.nan),
                        "Max Spread": annual_maxs.get(y, np.nan),
                        "Spread Range": annual_diffs.get(y, np.nan) # Added Metric
                    })
            
            progress_bar.progress((idx + 1) / len(pairs_items))
        
        st.session_state['annual_stats'] = pd.DataFrame(all_stats)

    if 'annual_stats' in st.session_state and not st.session_state['annual_stats'].empty:
        df_stats = st.session_state['annual_stats']
        
        metric_map = {
            "Average Spread": "Average Spread",
            "Spread Volatility (Std Dev)": "Spread Volatility",
            "Maximum Spread": "Max Spread",
            "Minimum Spread": "Min Spread",
            "Spread Range (Max - Min)": "Spread Range" # New Choice
        }
        
        col_ctrl, _ = st.columns([1, 2])
        with col_ctrl:
            selected_metric_label = st.radio("Select Metric to Visualize", list(metric_map.keys()), horizontal=True)
            selected_metric_col = metric_map[selected_metric_label]

        pivot_df = df_stats.pivot(index='Pair', columns='Year', values=selected_metric_col)
        
        if not pivot_df.empty:
            last_year = pivot_df.columns.max()
            pivot_df = pivot_df.sort_values(by=last_year, ascending=False)
        
        st.write(f"### Heatmap: {selected_metric_label}")
        
        # Color Map Logic
        if "Volatility" in selected_metric_label or "Range" in selected_metric_label:
            fmt = "{:.2f}"
            cmap = "Reds"
        else:
            fmt = "{:.1f}%"
            cmap = "RdYlGn" if "Min" not in selected_metric_label else "RdYlGn_r"
            
        st.dataframe(
            pivot_df.style
            .format(fmt, na_rep="")
            .background_gradient(cmap=cmap, axis=None)
            .highlight_null(props="background-color: white; color: black;"), 
            use_container_width=True,
            height=800
        )