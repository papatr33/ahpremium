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
    "Midea": {"A": "000333.SS", "H": "0300.HK"}
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

@st.cache_data(ttl=600)
def get_latest_spreads():
    """Sequential fetch of just the latest data (Stable)."""
    results = []
    start_date = date.today() - timedelta(days=10)
    end_date = date.today() + timedelta(days=1)
    
    for name, tickers in AH_PAIRS.items():
        df = fetch_pair_data(tickers['A'], tickers['H'], start_date, end_date)
        if not df.empty:
            df['A_USD'] = df['A_Local'] / df['USDCNH']
            df['H_USD'] = df['H_Local'] / df['USDHKD']
            current_spread = ( (df['A_USD'].iloc[-1] / df['H_USD'].iloc[-1]) - 1 ) * 100
            results.append({"Pair": name, "Current Spread (%)": current_spread})
    
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
start_date_input = st.sidebar.date_input("Start Date", date(2021, 1, 1))

# --- Main App ---
st.title(f"📉 AH Premium: Fixed Threshold Backtest")

tab1, tab2, tab3 = st.tabs(["Annual Stats Analysis", "Single Pair Analysis", "Rolling Correlations"])

# ==========================================
# TAB 1: Annual Stats Analysis
# ==========================================
with tab1:
    st.subheader("📊 Annual Spread Statistics & Comparison")
    
    with st.spinner("Scanning current spreads..."):
        latest_spread_df = get_latest_spreads()
    
    # Filter Logic: Auto-select pairs with spread < 10%
    default_selection = []
    if not latest_spread_df.empty:
        low_spread_pairs = latest_spread_df[latest_spread_df['Current Spread (%)'] < 10.0]
        default_selection = low_spread_pairs['Pair'].tolist()
        if not default_selection:
            default_selection = [list(AH_PAIRS.keys())[0]]
            
    # --- CHART SECTION ---
    st.write("#### 1. Spread History Comparison")
    st.caption("Default selection: Pairs with Current Spread < 10%")
    
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
    c_tbl, c_heat = st.columns([1, 2])
    with c_tbl:
        st.write("#### 2. Current Spread Snapshot")
        if not latest_spread_df.empty:
            st.dataframe(
                latest_spread_df.sort_values(by="Current Spread (%)").style.format({"Current Spread (%)": "{:.2f}%"}).background_gradient(cmap="RdYlGn_r"),
                use_container_width=True, height=600, hide_index=True
            )
        else:
            st.warning("Could not fetch latest spreads.")

    # --- HEATMAP SECTION ---
    with c_heat:
        st.write("#### 3. Annual Heatmap Generator")
        st.info("Click below to calculate historical stats.")
        
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
                    annual_diffs = annual_maxs - annual_mins
                    
                    unique_years = df_curr['Year'].unique()
                    for y in unique_years:
                        all_stats.append({
                            "Pair": name, "Year": y,
                            "Average Spread": annual_means.get(y, np.nan),
                            "Spread Volatility": annual_stds.get(y, np.nan),
                            "Min Spread": annual_mins.get(y, np.nan),
                            "Max Spread": annual_maxs.get(y, np.nan),
                            "Spread Range": annual_diffs.get(y, np.nan)
                        })
                progress_bar.progress((idx + 1) / len(pairs_items))
            st.session_state['annual_stats'] = pd.DataFrame(all_stats)

        if 'annual_stats' in st.session_state and not st.session_state['annual_stats'].empty:
            df_stats = st.session_state['annual_stats']
            metric_map = {
                "Average Spread": "Average Spread",
                "Spread Volatility": "Spread Volatility",
                "Maximum Spread": "Max Spread",
                "Minimum Spread": "Min Spread",
                "Spread Range (Max - Min)": "Spread Range"
            }
            selected_metric_label = st.radio("Select Metric", list(metric_map.keys()), horizontal=True)
            selected_metric_col = metric_map[selected_metric_label]

            pivot_df = df_stats.pivot(index='Pair', columns='Year', values=selected_metric_col)
            if not pivot_df.empty:
                last_year = pivot_df.columns.max()
                pivot_df = pivot_df.sort_values(by=last_year, ascending=False)
            
            if "Volatility" in selected_metric_label or "Range" in selected_metric_label:
                fmt = "{:.2f}"; cmap = "Reds"
            else:
                fmt = "{:.1f}%"; cmap = "RdYlGn" if "Min" not in selected_metric_label else "RdYlGn_r"
                
            st.dataframe(
                pivot_df.style.format(fmt, na_rep="").background_gradient(cmap=cmap, axis=None).highlight_null(props="background-color: white; color: black;"), 
                use_container_width=True, height=600
            )

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
# TAB 3: Rolling Correlations (NEW)
# ==========================================
with tab3:
    st.subheader("🔄 Rolling Correlation Analysis")
    st.caption("Analyze how A-Shares, H-Shares, and the Premium (Spread) move relative to each other over time.")
    
    col_corr_sel, _ = st.columns([1, 2])
    with col_corr_sel:
        # Unique key to avoid conflict with Tab 2
        selected_pair_corr = st.selectbox("Select AH Pair", list(AH_PAIRS.keys()), key="corr_pair_sel")
    
    tickers_corr = AH_PAIRS[selected_pair_corr]
    
    with st.spinner(f"Calculating correlations for {selected_pair_corr}..."):
        df_corr = fetch_pair_data(tickers_corr['A'], tickers_corr['H'], start_date_input, date.today())
        
        if not df_corr.empty:
            # 1. Data Prep: Calculate USD Returns and Spread Changes
            df_corr['A_USD'] = df_corr['A_Local'] / df_corr['USDCNH']
            df_corr['H_USD'] = df_corr['H_Local'] / df_corr['USDHKD']
            df_corr['Spread_Pct'] = ( (df_corr['A_USD'] / df_corr['H_USD']) - 1 ) * 100
            
            # Returns (1-day pct change)
            df_corr['Ret_A'] = df_corr['A_USD'].pct_change()
            df_corr['Ret_H'] = df_corr['H_USD'].pct_change()
            df_corr['Ret_Spread'] = df_corr['Spread_Pct'].diff() # Change in spread points
            
            # 2. Rolling Correlations
            # 30-Day
            df_corr['Corr30_AH'] = df_corr['Ret_A'].rolling(30).corr(df_corr['Ret_H'])
            df_corr['Corr30_SprA'] = df_corr['Ret_Spread'].rolling(30).corr(df_corr['Ret_A'])
            df_corr['Corr30_SprH'] = df_corr['Ret_Spread'].rolling(30).corr(df_corr['Ret_H'])
            
            # 60-Day
            df_corr['Corr60_AH'] = df_corr['Ret_A'].rolling(60).corr(df_corr['Ret_H'])
            df_corr['Corr60_SprA'] = df_corr['Ret_Spread'].rolling(60).corr(df_corr['Ret_A'])
            df_corr['Corr60_SprH'] = df_corr['Ret_Spread'].rolling(60).corr(df_corr['Ret_H'])
            
            # 3. Plots
            
            # Chart 1: A vs H Returns Correlation
            fig_ah = go.Figure()
            fig_ah.add_trace(go.Scatter(x=df_corr.index, y=df_corr['Corr30_AH'], name='30-Day Correlation', line=dict(color='#2A5CAA', width=1.5))) # Ruri
            fig_ah.add_trace(go.Scatter(x=df_corr.index, y=df_corr['Corr60_AH'], name='60-Day Correlation', line=dict(color='#9E3D3F', width=1.5))) # Suoh
            fig_ah.update_layout(
                title=f"Correlation: A-Share Returns vs H-Share Returns",
                yaxis_title="Correlation",
                template="seaborn", height=400, hovermode="x unified"
            )
            st.plotly_chart(fig_ah, use_container_width=True)
            
            c_c1, c_c2 = st.columns(2)
            
            # Chart 2: Spread vs A Returns
            with c_c1:
                fig_sa = go.Figure()
                fig_sa.add_trace(go.Scatter(x=df_corr.index, y=df_corr['Corr30_SprA'], name='30-Day', line=dict(color='#838B0D', width=1))) # Koke
                fig_sa.add_trace(go.Scatter(x=df_corr.index, y=df_corr['Corr60_SprA'], name='60-Day', line=dict(color='#FFB11B', width=1))) # Yamabuki
                fig_sa.update_layout(title="Correlation: Spread Change vs A-Share Returns", template="seaborn", height=350)
                st.plotly_chart(fig_sa, use_container_width=True)

            # Chart 3: Spread vs H Returns
            with c_c2:
                fig_sh = go.Figure()
                fig_sh.add_trace(go.Scatter(x=df_corr.index, y=df_corr['Corr30_SprH'], name='30-Day', line=dict(color='#838B0D', width=1)))
                fig_sh.add_trace(go.Scatter(x=df_corr.index, y=df_corr['Corr60_SprH'], name='60-Day', line=dict(color='#FFB11B', width=1)))
                fig_sh.update_layout(title="Correlation: Spread Change vs H-Share Returns", template="seaborn", height=350)
                st.plotly_chart(fig_sh, use_container_width=True)
        else:
            st.error("No data available for correlation analysis.")

