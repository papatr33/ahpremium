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

# --- Nippon Colors Palette (https://nipponcolors.com/) ---
# Traditional Japanese colors for data visualization
NIPPON_COLORS = [
    '#9E3D3F',  # Suoh (蘇芳) - Deep burgundy red
    '#005CAF',  # Ruri (瑠璃) - Lapis lazuli blue
    '#838B0D',  # Koke (苔) - Moss green
    '#FFB11B',  # Yamabuki (山吹) - Golden yellow
    '#6A4C9C',  # Sumire (菫) - Violet
    '#F17C67',  # Sangosyu (珊瑚朱) - Coral red
    '#3A8FB7',  # Hanada (縹) - Light blue
    '#6A8372',  # Byakuroku (白緑) - Pale green
    '#824880',  # Ayame (菖蒲) - Iris purple
    '#E16B8C',  # Kohbai (紅梅) - Pink plum
    '#1B813E',  # Wakatake (若竹) - Young bamboo green
]

# Named colors for specific UI elements
NIPPON_SUOH = '#9E3D3F'       # Deep red - for A-share
NIPPON_RURI = '#005CAF'       # Deep blue - for H-share  
NIPPON_KOKE = '#838B0D'       # Moss green - for Spread
NIPPON_KOHBAI = '#E16B8C'     # Pink plum - accent
NIPPON_WASURENAGUSA = '#7DB9DE'  # Forget-me-not blue - light accent

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

# --- Sidebar ---
st.sidebar.header("Settings")
start_date_input = st.sidebar.date_input("Start Date", date(2024, 1, 1))

# --- Main App ---
st.title(f"📉 AH Premium Viewer")

tab1, tab2, tab3, tab4 = st.tabs(["Spread Overview", "Pair Detail Chart", "Spread Comparison", "Correlation Analysis"])

# ==========================================
# TAB 1: Spread Overview
# ==========================================
with tab1:
    st.subheader("📊 Current Spread Snapshot")
    
    with st.spinner("Scanning current spreads..."):
        latest_spread_df = get_latest_spreads()
    
    # --- CURRENT SPREAD TABLE ---
    if not latest_spread_df.empty:
        st.dataframe(
            latest_spread_df.sort_values(by="Current Spread (%)").style.format({
                "Current Spread (%)": "{:.2f}%",
                "1D Change (%)": "{:.2f}%",
                "5D Change (%)": "{:.2f}%",
                "30D Change (%)": "{:.2f}%"
            }).background_gradient(cmap="RdYlGn_r", subset=["Current Spread (%)"]),
            use_container_width=True, height=700, hide_index=True
        )
    else:
        st.warning("Could not fetch latest spreads.")

# ==========================================
# TAB 2: Pair Detail Chart (A, H prices + Spread)
# ==========================================
with tab2:
    st.subheader("📈 Individual Pair Analysis")
    
    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        selected_pair = st.selectbox("Select AH Pair", list(AH_PAIRS.keys()), key="detail_pair_sel")
    
    pair_tickers = AH_PAIRS[selected_pair]
    
    with col_info:
        st.caption(f"A-Share: `{pair_tickers['A']}` | H-Share: `{pair_tickers['H']}`")

    with st.spinner(f"Loading data for {selected_pair}..."):
        raw_data = fetch_pair_data(pair_tickers['A'], pair_tickers['H'], start_date_input, date.today())
        
        if raw_data.empty:
            st.error("No data found for this pair.")
        else:
            # Calculate USD prices and spread
            df_view = raw_data.copy()
            df_view['A_USD'] = df_view['A_Local'] / df_view['USDCNH']
            df_view['H_USD'] = df_view['H_Local'] / df_view['USDHKD']
            df_view['Spread_Pct'] = ((df_view['A_USD'] / df_view['H_USD']) - 1) * 100
            
            # Current metrics
            current_spread = df_view['Spread_Pct'].iloc[-1]
            current_a = df_view['A_Local'].iloc[-1]
            current_h = df_view['H_Local'].iloc[-1]
            spread_avg = df_view['Spread_Pct'].mean()
            spread_std = df_view['Spread_Pct'].std()
            spread_z = (current_spread - spread_avg) / spread_std if spread_std > 0 else 0
            
            # Metrics row
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Current Spread", f"{current_spread:.2f}%")
            c2.metric("A-Share Price (CNY)", f"¥{current_a:.2f}")
            c3.metric("H-Share Price (HKD)", f"HK${current_h:.2f}")
            c4.metric("Spread Avg", f"{spread_avg:.2f}%")
            c5.metric("Z-Score", f"{spread_z:.2f}")
            
            st.divider()
            
            # --- Main Chart: A & H Prices with Spread ---
            # Using Nippon Colors: Suoh (red) for A, Ruri (blue) for H, Koke (green) for Spread
            SUOH = NIPPON_SUOH
            RURI = NIPPON_RURI
            KOKE = NIPPON_KOKE
            
            fig_detail = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.08, 
                row_heights=[0.6, 0.4],
                subplot_titles=("A-Share & H-Share Prices (USD)", "AH Spread (%)")
            )
            
            # Row 1: A and H prices in USD
            fig_detail.add_trace(
                go.Scatter(
                    x=df_view.index, 
                    y=df_view['A_USD'], 
                    name='A-Share (USD)', 
                    line=dict(color=SUOH, width=2),
                    hovertemplate='%{y:.2f}<extra>A-Share (USD)</extra>'
                ), 
                row=1, col=1
            )
            fig_detail.add_trace(
                go.Scatter(
                    x=df_view.index, 
                    y=df_view['H_USD'], 
                    name='H-Share (USD)', 
                    line=dict(color=RURI, width=2),
                    hovertemplate='%{y:.2f}<extra>H-Share (USD)</extra>'
                ), 
                row=1, col=1
            )
            
            # Row 2: Spread
            fig_detail.add_trace(
                go.Scatter(
                    x=df_view.index, 
                    y=df_view['Spread_Pct'], 
                    name='Spread (%)', 
                    fill='tozeroy',
                    line=dict(color=KOKE, width=1.5),
                    fillcolor='rgba(131, 139, 13, 0.3)',
                    hovertemplate='%{y:.2f}%<extra>Spread</extra>'
                ), 
                row=2, col=1
            )
            
            # Add average spread line
            fig_detail.add_hline(
                y=spread_avg, 
                line_dash="dash", 
                line_color="gray", 
                row=2, col=1
            )
            
            # Add ±1 std bands
            fig_detail.add_hline(y=spread_avg + spread_std, line_dash="dot", line_color="lightgray", row=2, col=1)
            fig_detail.add_hline(y=spread_avg - spread_std, line_dash="dot", line_color="lightgray", row=2, col=1)
            
            fig_detail.update_layout(
                height=650, 
                template="seaborn", 
                hovermode="x unified",
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_detail.update_yaxes(title_text="Price (USD)", row=1, col=1)
            fig_detail.update_yaxes(title_text="Spread (%)", row=2, col=1)
            
            st.plotly_chart(fig_detail, use_container_width=True)
            
            # --- Secondary Chart: Local Currency Prices ---
            with st.expander("📊 View Local Currency Prices (CNY & HKD)"):
                fig_local = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("A-Share Price (CNY)", "H-Share Price (HKD)")
                )
                
                fig_local.add_trace(
                    go.Scatter(
                        x=df_view.index, 
                        y=df_view['A_Local'], 
                        name='A (CNY)',
                        line=dict(color=SUOH, width=1.5),
                        hovertemplate='%{y:.2f}<extra>A (CNY)</extra>'
                    ),
                    row=1, col=1
                )
                fig_local.add_trace(
                    go.Scatter(
                        x=df_view.index, 
                        y=df_view['H_Local'], 
                        name='H (HKD)',
                        line=dict(color=RURI, width=1.5),
                        hovertemplate='%{y:.2f}<extra>H (HKD)</extra>'
                    ),
                    row=1, col=2
                )
                
                fig_local.update_layout(height=350, template="seaborn", showlegend=False, hovermode="x unified")
                st.plotly_chart(fig_local, use_container_width=True)
            
            # --- Statistics Table ---
            with st.expander("📈 Spread Statistics"):
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    stats_df = pd.DataFrame({
                        'Metric': ['Current', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th Pct', '75th Pct'],
                        'Spread (%)': [
                            current_spread,
                            spread_avg,
                            df_view['Spread_Pct'].median(),
                            spread_std,
                            df_view['Spread_Pct'].min(),
                            df_view['Spread_Pct'].max(),
                            df_view['Spread_Pct'].quantile(0.25),
                            df_view['Spread_Pct'].quantile(0.75)
                        ]
                    })
                    st.dataframe(
                        stats_df.style.format({'Spread (%)': '{:.2f}'}),
                        use_container_width=True, 
                        hide_index=True
                    )
                
                with col_stat2:
                    # Spread distribution histogram
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=df_view['Spread_Pct'],
                        nbinsx=40,
                        marker_color=KOKE,
                        opacity=0.7,
                        name='Spread Distribution',
                        hovertemplate='Spread: %{x:.2f}%<br>Count: %{y}<extra></extra>'
                    ))
                    fig_hist.add_vline(x=current_spread, line_dash="solid", line_color=SUOH, 
                                       annotation_text=f"Current: {current_spread:.2f}%")
                    fig_hist.add_vline(x=spread_avg, line_dash="dash", line_color="gray")
                    fig_hist.update_layout(
                        height=280,
                        template="seaborn",
                        showlegend=False,
                        xaxis_title="Spread (%)",
                        yaxis_title="Frequency"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# TAB 3: Spread Comparison
# ==========================================
with tab3:
    st.subheader("📊 Multi-Pair Spread Comparison")
    
    # User input for threshold
    spread_threshold = st.number_input(
        "Default selection threshold: Pairs with Current Spread < (%)", 
        value=10.0, 
        step=1.0,
        help="Pairs with current spread below this threshold will be auto-selected",
        key="spread_threshold_tab3"
    )
    
    # Get latest spreads for filtering
    with st.spinner("Scanning spreads..."):
        latest_spread_df_t3 = get_latest_spreads()
    
    # Filter Logic: Auto-select pairs with spread < user-defined threshold
    default_selection = []
    if not latest_spread_df_t3.empty:
        low_spread_pairs = latest_spread_df_t3[latest_spread_df_t3['Current Spread (%)'] < spread_threshold]
        default_selection = low_spread_pairs['Pair'].tolist()
        if not default_selection:
            default_selection = [list(AH_PAIRS.keys())[0]]
    
    selected_chart_pairs = st.multiselect(
        "Select Pairs to Compare", 
        options=list(AH_PAIRS.keys()),
        default=default_selection,
        key="compare_pairs_tab3"
    )
    
    if selected_chart_pairs:
        fig_comp = go.Figure()
        for i, p in enumerate(selected_chart_pairs):
            p_tickers = AH_PAIRS[p]
            df_p = fetch_pair_data(p_tickers['A'], p_tickers['H'], start_date_input, date.today())
            if not df_p.empty:
                df_p['A_USD'] = df_p['A_Local'] / df_p['USDCNH']
                df_p['H_USD'] = df_p['H_Local'] / df_p['USDHKD']
                spread_series = ((df_p['A_USD'] / df_p['H_USD']) - 1) * 100
                color_hex = NIPPON_COLORS[i % len(NIPPON_COLORS)]
                fig_comp.add_trace(go.Scatter(
                    x=df_p.index, 
                    y=spread_series, 
                    name=p, 
                    line=dict(color=color_hex, width=1.5)
                ))
        
        fig_comp.update_layout(
            title="Historical Spread (%) Comparison",
            template="seaborn", 
            hovermode="x unified", 
            height=550,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        fig_comp.update_yaxes(title_text="Spread (%)")
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Select at least one pair to view the comparison chart.")

# ==========================================
# TAB 4: Correlation Analysis (ENHANCED)
# ==========================================
with tab4:
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
                                        name=f'{rolling_window}-Day Corr', line=dict(color=NIPPON_RURI, width=2)))
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
                                           name='Corr(Spread, A)', line=dict(color=NIPPON_KOKE, width=1.5)))
                fig_sa.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_sa.update_layout(title="Spread Change vs A-Share Returns", 
                                    yaxis_range=[-1, 1], template="seaborn", height=300)
                st.plotly_chart(fig_sa, use_container_width=True)
            
            with c_c2:
                fig_sh = go.Figure()
                fig_sh.add_trace(go.Scatter(x=df_corr.index, y=df_corr['Roll_Corr_SprH'], 
                                           name='Corr(Spread, H)', line=dict(color=NIPPON_KOHBAI, width=1.5)))
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
                fig_lag1.add_trace(go.Bar(x=lags, y=cross_corr_ah, marker_color=NIPPON_RURI, name='Cross-Corr'))
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
                fig_lag2.add_trace(go.Bar(x=lags, y=cross_corr_a_spread, marker_color=NIPPON_SUOH, name='Cross-Corr'))
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
            # SECTION 4: CONDITIONAL OUTPERFORMANCE ANALYSIS
            # ===========================================
            st.markdown("---")
            st.markdown("### 🎯 Conditional Outperformance Analysis")
            st.caption("When both markets move in the same direction, who tends to outperform?")
            
            # Calculate outperformance
            df_clean['A_Outperformance'] = df_clean['Ret_A'] - df_clean['Ret_H']
            
            # Define regimes
            both_up = (df_clean['Ret_A'] > 0) & (df_clean['Ret_H'] > 0)
            both_down = (df_clean['Ret_A'] < 0) & (df_clean['Ret_H'] < 0)
            a_up_h_down = (df_clean['Ret_A'] > 0) & (df_clean['Ret_H'] < 0)
            a_down_h_up = (df_clean['Ret_A'] < 0) & (df_clean['Ret_H'] > 0)
            
            df_both_up = df_clean[both_up]
            df_both_down = df_clean[both_down]
            df_a_up_h_down = df_clean[a_up_h_down]
            df_a_down_h_up = df_clean[a_down_h_up]
            
            # Calculate statistics for each regime
            regimes_data = []
            
            if len(df_both_up) > 0:
                avg_outperf_up = df_both_up['A_Outperformance'].mean()
                win_rate_up = (df_both_up['A_Outperformance'] > 0).mean() * 100
                t_stat_up, p_val_up = stats.ttest_1samp(df_both_up['A_Outperformance'], 0)
                regimes_data.append({
                    'Regime': '📈 Both Up',
                    'Days': len(df_both_up),
                    'Pct of Total': f"{len(df_both_up)/len(df_clean)*100:.1f}%",
                    'Avg A Outperformance (%)': avg_outperf_up,
                    'A Wins (%)': win_rate_up,
                    'p-value': p_val_up,
                    'Significant': '✓' if p_val_up < 0.05 else '✗'
                })
            
            if len(df_both_down) > 0:
                avg_outperf_down = df_both_down['A_Outperformance'].mean()
                win_rate_down = (df_both_down['A_Outperformance'] > 0).mean() * 100
                t_stat_down, p_val_down = stats.ttest_1samp(df_both_down['A_Outperformance'], 0)
                regimes_data.append({
                    'Regime': '📉 Both Down',
                    'Days': len(df_both_down),
                    'Pct of Total': f"{len(df_both_down)/len(df_clean)*100:.1f}%",
                    'Avg A Outperformance (%)': avg_outperf_down,
                    'A Wins (%)': win_rate_down,
                    'p-value': p_val_down,
                    'Significant': '✓' if p_val_down < 0.05 else '✗'
                })
            
            if len(df_a_up_h_down) > 0:
                regimes_data.append({
                    'Regime': '🔀 A Up, H Down',
                    'Days': len(df_a_up_h_down),
                    'Pct of Total': f"{len(df_a_up_h_down)/len(df_clean)*100:.1f}%",
                    'Avg A Outperformance (%)': df_a_up_h_down['A_Outperformance'].mean(),
                    'A Wins (%)': 100.0,  # By definition
                    'p-value': np.nan,
                    'Significant': 'N/A'
                })
            
            if len(df_a_down_h_up) > 0:
                regimes_data.append({
                    'Regime': '🔀 A Down, H Up',
                    'Days': len(df_a_down_h_up),
                    'Pct of Total': f"{len(df_a_down_h_up)/len(df_clean)*100:.1f}%",
                    'Avg A Outperformance (%)': df_a_down_h_up['A_Outperformance'].mean(),
                    'A Wins (%)': 0.0,  # By definition
                    'p-value': np.nan,
                    'Significant': 'N/A'
                })
            
            regime_summary_df = pd.DataFrame(regimes_data)
            
            # Display summary table
            col_regime_tbl, col_regime_chart = st.columns([1, 1])
            
            with col_regime_tbl:
                st.markdown("#### Summary by Market Regime")
                st.dataframe(
                    regime_summary_df.style.format({
                        'Avg A Outperformance (%)': '{:.3f}',
                        'A Wins (%)': '{:.1f}',
                        'p-value': '{:.4f}'
                    }).background_gradient(
                        cmap="RdYlGn", 
                        subset=['Avg A Outperformance (%)'],
                        vmin=-0.5, vmax=0.5
                    ),
                    use_container_width=True, hide_index=True
                )
                
                # Key insight
                if len(df_both_up) > 0 and len(df_both_down) > 0:
                    avg_up = df_both_up['A_Outperformance'].mean()
                    avg_down = df_both_down['A_Outperformance'].mean()
                    
                    st.markdown("#### 💡 Key Insights")
                    
                    if avg_up > 0 and p_val_up < 0.05:
                        st.success(f"**Bull Days:** A-shares outperform H by **{avg_up:.3f}%** on average (statistically significant)")
                    elif avg_up < 0 and p_val_up < 0.05:
                        st.error(f"**Bull Days:** H-shares outperform A by **{-avg_up:.3f}%** on average (statistically significant)")
                    else:
                        st.info(f"**Bull Days:** Average A outperformance is {avg_up:.3f}% (not statistically significant)")
                    
                    if avg_down > 0 and p_val_down < 0.05:
                        st.success(f"**Bear Days:** A-shares outperform (fall less) by **{avg_down:.3f}%** on average (statistically significant)")
                    elif avg_down < 0 and p_val_down < 0.05:
                        st.error(f"**Bear Days:** H-shares outperform (fall less) by **{-avg_down:.3f}%** on average (statistically significant)")
                    else:
                        st.info(f"**Bear Days:** Average A outperformance is {avg_down:.3f}% (not statistically significant)")
            
            with col_regime_chart:
                # Quadrant scatter plot
                fig_quadrant = go.Figure()
                
                # Color by regime
                colors_regime = []
                for i in range(len(df_clean)):
                    if df_clean['Ret_A'].iloc[i] > 0 and df_clean['Ret_H'].iloc[i] > 0:
                        colors_regime.append('#2E7D32')  # Green - both up
                    elif df_clean['Ret_A'].iloc[i] < 0 and df_clean['Ret_H'].iloc[i] < 0:
                        colors_regime.append('#C62828')  # Red - both down
                    else:
                        colors_regime.append('#757575')  # Gray - divergent
                
                fig_quadrant.add_trace(go.Scatter(
                    x=df_clean['Ret_H'], y=df_clean['Ret_A'],
                    mode='markers',
                    marker=dict(color=colors_regime, size=6, opacity=0.6),
                    text=[f"A: {a:.2f}%, H: {h:.2f}%<br>A-H: {o:.2f}%" 
                          for a, h, o in zip(df_clean['Ret_A'], df_clean['Ret_H'], df_clean['A_Outperformance'])],
                    hoverinfo='text',
                    name='Daily Returns'
                ))
                
                # Add diagonal line (A = H)
                max_val = max(abs(df_clean['Ret_A'].max()), abs(df_clean['Ret_H'].max()),
                             abs(df_clean['Ret_A'].min()), abs(df_clean['Ret_H'].min()))
                fig_quadrant.add_trace(go.Scatter(
                    x=[-max_val, max_val], y=[-max_val, max_val],
                    mode='lines', line=dict(color='gray', dash='dash', width=1),
                    name='A = H line'
                ))
                
                # Add quadrant lines
                fig_quadrant.add_hline(y=0, line_color="black", line_width=0.5)
                fig_quadrant.add_vline(x=0, line_color="black", line_width=0.5)
                
                fig_quadrant.update_layout(
                    title="Return Quadrant Analysis<br><sub>Green=Both Up, Red=Both Down, Gray=Divergent</sub>",
                    xaxis_title="H-Share Return (%)",
                    yaxis_title="A-Share Return (%)",
                    template="seaborn", height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_quadrant, use_container_width=True)
            
            # Distribution of outperformance by regime
            st.markdown("#### Distribution of A-Share Outperformance by Regime")
            
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                if len(df_both_up) > 5:
                    fig_dist_up = go.Figure()
                    fig_dist_up.add_trace(go.Histogram(
                        x=df_both_up['A_Outperformance'], 
                        nbinsx=30,
                        marker_color='#2E7D32',
                        name='Both Up Days'
                    ))
                    fig_dist_up.add_vline(x=0, line_dash="dash", line_color="black")
                    fig_dist_up.add_vline(x=df_both_up['A_Outperformance'].mean(), 
                                         line_dash="solid", line_color="red",
                                         annotation_text=f"Mean: {df_both_up['A_Outperformance'].mean():.3f}%")
                    fig_dist_up.update_layout(
                        title="📈 Both Up Days: A Outperformance Distribution",
                        xaxis_title="A Return - H Return (%)",
                        yaxis_title="Frequency",
                        template="seaborn", height=300
                    )
                    st.plotly_chart(fig_dist_up, use_container_width=True)
            
            with col_dist2:
                if len(df_both_down) > 5:
                    fig_dist_down = go.Figure()
                    fig_dist_down.add_trace(go.Histogram(
                        x=df_both_down['A_Outperformance'], 
                        nbinsx=30,
                        marker_color='#C62828',
                        name='Both Down Days'
                    ))
                    fig_dist_down.add_vline(x=0, line_dash="dash", line_color="black")
                    fig_dist_down.add_vline(x=df_both_down['A_Outperformance'].mean(), 
                                           line_dash="solid", line_color="blue",
                                           annotation_text=f"Mean: {df_both_down['A_Outperformance'].mean():.3f}%")
                    fig_dist_down.update_layout(
                        title="📉 Both Down Days: A Outperformance Distribution",
                        xaxis_title="A Return - H Return (%)",
                        yaxis_title="Frequency",
                        template="seaborn", height=300
                    )
                    st.plotly_chart(fig_dist_down, use_container_width=True)
            
            # Trading implication
            with st.expander("📖 Trading Implications"):
                st.markdown("""
                **How to interpret this analysis:**
                
                1. **If A outperforms on Bull Days (Both Up):**
                   - A-shares have higher beta to positive sentiment
                   - In rallies, A-shares tend to rise MORE than H-shares
                   - Spread WIDENS during bull markets
                   
                2. **If H outperforms on Bull Days:**
                   - H-shares capture more upside
                   - Spread NARROWS during bull markets
                   - Potential for spread compression trades
                
                3. **If A outperforms on Bear Days (Both Down):**
                   - A-shares are more defensive (fall less)
                   - Spread WIDENS during selloffs
                   - A-shares provide relative downside protection
                
                4. **If H outperforms on Bear Days:**
                   - H-shares fall less in corrections
                   - Spread NARROWS during selloffs
                   - H-shares are the defensive leg
                
                **Strategic Use:**
                - If A consistently outperforms in BOTH regimes → structural A premium justified
                - If H outperforms in bull, A outperforms in bear → spread mean-reverts
                - Asymmetric patterns suggest tactical timing opportunities
                """)
            
            # ===========================================
            # SECTION 4B: MULTI-HORIZON OUTPERFORMANCE (NON-OVERLAPPING)
            # ===========================================
            st.markdown("---")
            st.markdown("### 📅 Multi-Horizon Conditional Outperformance (Non-Overlapping Periods)")
            st.caption("Slicing data into distinct, non-overlapping periods for statistically valid analysis")
            
            # User selectable period length
            col_period_select, col_analysis_mode = st.columns(2)
            with col_period_select:
                period_options = [5, 10, 20, 60]
                selected_period = st.selectbox(
                    "Period length (trading days)", 
                    options=period_options,
                    index=2,  # Default to 20
                    help="Data will be sliced into non-overlapping chunks of this size"
                )
            with col_analysis_mode:
                show_all_periods = st.checkbox("Show all individual periods", value=False,
                                              help="Display each period's data in a table")
            
            # Slice data into non-overlapping periods
            df_periods = df_clean.copy()
            n_periods = len(df_periods) // selected_period
            
            if n_periods >= 3:
                # Create period chunks
                period_data = []
                
                for i in range(n_periods):
                    start_idx = i * selected_period
                    end_idx = (i + 1) * selected_period
                    chunk = df_periods.iloc[start_idx:end_idx]
                    
                    if len(chunk) == selected_period:
                        period_start = chunk.index[0]
                        period_end = chunk.index[-1]
                        
                        # Calculate cumulative returns for this period
                        cum_a = chunk['Ret_A'].sum()
                        cum_h = chunk['Ret_H'].sum()
                        outperf = cum_a - cum_h
                        
                        # Classify regime
                        if cum_a > 0 and cum_h > 0:
                            regime = '📈 Both Up'
                        elif cum_a < 0 and cum_h < 0:
                            regime = '📉 Both Down'
                        elif cum_a > 0 and cum_h < 0:
                            regime = '🔀 A Up, H Down'
                        else:
                            regime = '🔀 A Down, H Up'
                        
                        period_data.append({
                            'Period #': i + 1,
                            'Start': period_start,
                            'End': period_end,
                            'A Return (%)': cum_a,
                            'H Return (%)': cum_h,
                            'A Outperformance (%)': outperf,
                            'Regime': regime
                        })
                
                periods_df = pd.DataFrame(period_data)
                
                st.info(f"📊 Data sliced into **{len(periods_df)} non-overlapping {selected_period}-day periods** (from {periods_df['Start'].iloc[0].strftime('%Y-%m-%d')} to {periods_df['End'].iloc[-1].strftime('%Y-%m-%d')})")
                
                # Summary statistics by regime
                st.markdown("#### Summary by Regime (Non-Overlapping Periods)")
                
                regime_summary = []
                for regime in ['📈 Both Up', '📉 Both Down', '🔀 A Up, H Down', '🔀 A Down, H Up']:
                    regime_df = periods_df[periods_df['Regime'] == regime]
                    if len(regime_df) >= 1:
                        outperf_series = regime_df['A Outperformance (%)']
                        avg_a = regime_df['A Return (%)'].mean()
                        avg_h = regime_df['H Return (%)'].mean()
                        avg_outperf = outperf_series.mean()
                        win_rate = (outperf_series > 0).mean() * 100
                        
                        # Statistical test (only if enough observations)
                        if len(regime_df) >= 3:
                            t_stat, p_val = stats.ttest_1samp(outperf_series, 0)
                        else:
                            p_val = np.nan
                        
                        regime_summary.append({
                            'Regime': regime,
                            'Periods': len(regime_df),
                            'Avg A Return (%)': avg_a,
                            'Avg H Return (%)': avg_h,
                            'Avg A Outperf (%)': avg_outperf,
                            'A Wins (%)': win_rate,
                            'p-value': p_val,
                            'Sig': '✓' if p_val < 0.05 else ('—' if np.isnan(p_val) else '')
                        })
                
                summary_df = pd.DataFrame(regime_summary)
                
                col_summary, col_chart = st.columns([1, 1])
                
                with col_summary:
                    st.dataframe(
                        summary_df.style.format({
                            'Avg A Return (%)': '{:.2f}',
                            'Avg H Return (%)': '{:.2f}',
                            'Avg A Outperf (%)': '{:.2f}',
                            'A Wins (%)': '{:.1f}',
                            'p-value': '{:.4f}'
                        }).background_gradient(
                            cmap="RdYlGn",
                            subset=['Avg A Outperf (%)'],
                            vmin=-5, vmax=5
                        ),
                        use_container_width=True, hide_index=True
                    )
                    
                    # Note about sample size
                    co_movement_periods = summary_df[summary_df['Regime'].isin(['📈 Both Up', '📉 Both Down'])]['Periods'].sum()
                    st.caption(f"⚠️ {co_movement_periods} co-movement periods (Both Up + Both Down) — small samples require cautious interpretation")
                
                with col_chart:
                    # Scatter plot of all periods
                    fig_scatter_periods = go.Figure()
                    
                    colors_map = {
                        '📈 Both Up': '#2E7D32',
                        '📉 Both Down': '#C62828',
                        '🔀 A Up, H Down': '#1565C0',
                        '🔀 A Down, H Up': '#FF8F00'
                    }
                    
                    for regime in colors_map.keys():
                        regime_df = periods_df[periods_df['Regime'] == regime]
                        if len(regime_df) > 0:
                            fig_scatter_periods.add_trace(go.Scatter(
                                x=regime_df['H Return (%)'],
                                y=regime_df['A Return (%)'],
                                mode='markers',
                                name=regime,
                                marker=dict(color=colors_map[regime], size=10, opacity=0.7),
                                text=[f"Period {p}<br>{s.strftime('%m/%d')}-{e.strftime('%m/%d')}<br>A: {a:.1f}%, H: {h:.1f}%" 
                                      for p, s, e, a, h in zip(regime_df['Period #'], regime_df['Start'], 
                                                               regime_df['End'], regime_df['A Return (%)'], 
                                                               regime_df['H Return (%)'])],
                                hoverinfo='text'
                            ))
                    
                    # Add diagonal line
                    max_val = max(abs(periods_df['A Return (%)'].max()), abs(periods_df['H Return (%)'].max()),
                                 abs(periods_df['A Return (%)'].min()), abs(periods_df['H Return (%)'].min()))
                    fig_scatter_periods.add_trace(go.Scatter(
                        x=[-max_val*1.1, max_val*1.1], y=[-max_val*1.1, max_val*1.1],
                        mode='lines', line=dict(color='gray', dash='dash', width=1),
                        name='A = H', showlegend=False
                    ))
                    
                    fig_scatter_periods.add_hline(y=0, line_color="black", line_width=0.5)
                    fig_scatter_periods.add_vline(x=0, line_color="black", line_width=0.5)
                    
                    fig_scatter_periods.update_layout(
                        title=f"All {selected_period}-Day Periods<br><sub>Each dot = one non-overlapping period</sub>",
                        xaxis_title=f"H-Share {selected_period}D Return (%)",
                        yaxis_title=f"A-Share {selected_period}D Return (%)",
                        template="seaborn",
                        height=400
                    )
                    st.plotly_chart(fig_scatter_periods, use_container_width=True)
                
                # Key findings for co-movement regimes
                st.markdown("#### 💡 Key Findings (Co-Movement Periods Only)")
                
                both_up_df = periods_df[periods_df['Regime'] == '📈 Both Up']
                both_down_df = periods_df[periods_df['Regime'] == '📉 Both Down']
                
                col_up, col_down = st.columns(2)
                
                with col_up:
                    if len(both_up_df) >= 1:
                        avg_a = both_up_df['A Return (%)'].mean()
                        avg_h = both_up_df['H Return (%)'].mean()
                        avg_outperf = both_up_df['A Outperformance (%)'].mean()
                        win_rate = (both_up_df['A Outperformance (%)'] > 0).mean() * 100
                        
                        st.markdown(f"**📈 Bull Periods ({len(both_up_df)} periods)**")
                        st.markdown(f"- Avg A Return: **{avg_a:.2f}%**")
                        st.markdown(f"- Avg H Return: **{avg_h:.2f}%**")
                        
                        if avg_outperf > 0:
                            st.success(f"A outperforms by **{avg_outperf:.2f}%** on average")
                            st.markdown(f"A wins **{win_rate:.0f}%** of bull periods")
                        else:
                            st.error(f"H outperforms by **{-avg_outperf:.2f}%** on average")
                            st.markdown(f"A wins only **{win_rate:.0f}%** of bull periods")
                    else:
                        st.info("No 'Both Up' periods in this data range")
                
                with col_down:
                    if len(both_down_df) >= 1:
                        avg_a = both_down_df['A Return (%)'].mean()
                        avg_h = both_down_df['H Return (%)'].mean()
                        avg_outperf = both_down_df['A Outperformance (%)'].mean()
                        win_rate = (both_down_df['A Outperformance (%)'] > 0).mean() * 100
                        
                        st.markdown(f"**📉 Bear Periods ({len(both_down_df)} periods)**")
                        st.markdown(f"- Avg A Return: **{avg_a:.2f}%**")
                        st.markdown(f"- Avg H Return: **{avg_h:.2f}%**")
                        
                        if avg_outperf > 0:
                            st.success(f"A falls less by **{avg_outperf:.2f}%** on average")
                            st.markdown(f"A more defensive in **{win_rate:.0f}%** of bear periods")
                        else:
                            st.error(f"H falls less by **{-avg_outperf:.2f}%** on average")
                            st.markdown(f"A more defensive in only **{win_rate:.0f}%** of bear periods")
                    else:
                        st.info("No 'Both Down' periods in this data range")
                
                # Individual periods table
                if show_all_periods:
                    st.markdown("---")
                    st.markdown("#### All Individual Periods")
                    
                    display_df = periods_df.copy()
                    display_df['Start'] = display_df['Start'].dt.strftime('%Y-%m-%d')
                    display_df['End'] = display_df['End'].dt.strftime('%Y-%m-%d')
                    
                    st.dataframe(
                        display_df.style.format({
                            'A Return (%)': '{:.2f}',
                            'H Return (%)': '{:.2f}',
                            'A Outperformance (%)': '{:.2f}'
                        }).background_gradient(
                            cmap="RdYlGn",
                            subset=['A Outperformance (%)'],
                            vmin=-10, vmax=10
                        ),
                        use_container_width=True, hide_index=True,
                        height=400
                    )
                
                # Timeline visualization
                st.markdown("---")
                st.markdown("#### Period-by-Period Timeline")
                
                fig_bars = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                        subplot_titles=(f"{selected_period}-Day Period Returns", 
                                                       "A Outperformance (A - H)"))
                
                # Period returns
                fig_bars.add_trace(go.Bar(
                    x=periods_df['Period #'],
                    y=periods_df['A Return (%)'],
                    name='A Return',
                    marker_color=NIPPON_SUOH
                ), row=1, col=1)
                
                fig_bars.add_trace(go.Bar(
                    x=periods_df['Period #'],
                    y=periods_df['H Return (%)'],
                    name='H Return',
                    marker_color=NIPPON_RURI
                ), row=1, col=1)
                
                # Outperformance bars colored by who won
                outperf_colors = ['#2E7D32' if v > 0 else '#C62828' for v in periods_df['A Outperformance (%)']]
                fig_bars.add_trace(go.Bar(
                    x=periods_df['Period #'],
                    y=periods_df['A Outperformance (%)'],
                    name='A Outperf',
                    marker_color=outperf_colors,
                    text=[f"{v:.1f}%" for v in periods_df['A Outperformance (%)']],
                    textposition='outside'
                ), row=2, col=1)
                
                fig_bars.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
                fig_bars.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
                
                fig_bars.update_layout(
                    height=500, 
                    template="seaborn", 
                    barmode='group',
                    hovermode="x unified"
                )
                fig_bars.update_xaxes(title_text="Period #", row=2, col=1)
                
                st.plotly_chart(fig_bars, use_container_width=True)
                
                # Interpretation
                with st.expander("📖 Why non-overlapping periods matter"):
                    st.markdown(f"""
                    **The Problem with Rolling Windows:**
                    
                    If you use rolling 20-day windows on 120 trading days, you get ~100 "observations."
                    But these overlap heavily — window 1 (days 1-20) shares 19 days with window 2 (days 2-21).
                    
                    This causes:
                    - **Autocorrelation**: Observations are not independent
                    - **Inflated sample size**: You think you have 100 samples, but really ~6 independent periods
                    - **Invalid p-values**: t-tests assume independent observations
                    
                    **The Solution: Non-Overlapping Periods**
                    
                    Slicing into distinct chunks (days 1-20, 21-40, 41-60, etc.) gives you:
                    - **Independent observations**: Each period's data is unique
                    - **Valid statistics**: t-tests and confidence intervals are meaningful
                    - **Fewer but honest samples**: {len(periods_df)} true observations
                    
                    **Trade-off:**
                    - Fewer observations = wider confidence intervals
                    - But the conclusions are statistically valid
                    - Use longer date ranges to get more non-overlapping periods
                    """)
                
            else:
                st.warning(f"Need at least {selected_period * 3} days of data for meaningful analysis. Current data has {len(df_periods)} days.")
            
            # Keep rolling analysis as secondary option
            st.markdown("---")
            with st.expander("📊 Rolling Window Analysis (for reference - overlapping periods)"):
                st.caption("⚠️ Uses overlapping periods - statistics may be inflated. Use for pattern visualization, not statistical inference.")
                
                selected_lookbacks = [5, 20]
                
                # Calculate cumulative returns for each lookback
                for lb in selected_lookbacks:
                    df_clean[f'Cum_A_{lb}d'] = df_clean['Ret_A'].rolling(lb).sum()
                    df_clean[f'Cum_H_{lb}d'] = df_clean['Ret_H'].rolling(lb).sum()
                    df_clean[f'Cum_Outperf_{lb}d'] = df_clean[f'Cum_A_{lb}d'] - df_clean[f'Cum_H_{lb}d']
                
                # Build analysis for each lookback period
                multi_horizon_results = []
                
                for lb in selected_lookbacks:
                    df_lb = df_clean.dropna(subset=[f'Cum_A_{lb}d', f'Cum_H_{lb}d', f'Cum_Outperf_{lb}d'])
                    
                    if len(df_lb) < 20:
                        continue
                    
                    # Define regimes based on cumulative returns
                    both_up_lb = (df_lb[f'Cum_A_{lb}d'] > 0) & (df_lb[f'Cum_H_{lb}d'] > 0)
                    both_down_lb = (df_lb[f'Cum_A_{lb}d'] < 0) & (df_lb[f'Cum_H_{lb}d'] < 0)
                    
                    df_lb_up = df_lb[both_up_lb]
                    df_lb_down = df_lb[both_down_lb]
                    
                    # Both Up regime - who gained more?
                    if len(df_lb_up) >= 10:
                        outperf = df_lb_up[f'Cum_Outperf_{lb}d']
                        avg_outperf = outperf.mean()
                        avg_a = df_lb_up[f'Cum_A_{lb}d'].mean()
                        avg_h = df_lb_up[f'Cum_H_{lb}d'].mean()
                        win_rate = (outperf > 0).mean() * 100
                        t_stat, p_val = stats.ttest_1samp(outperf.dropna(), 0)
                        
                        multi_horizon_results.append({
                            'Period': f'{lb}D',
                            'Regime': '📈 Both Up',
                            'Observations': len(df_lb_up),
                            'Avg A Return (%)': avg_a,
                            'Avg H Return (%)': avg_h,
                            'A Outperformance (%)': avg_outperf,
                            'A Wins (%)': win_rate,
                            'p-value': p_val,
                            'Sig': '✓' if p_val < 0.05 else ''
                        })
                    
                    # Both Down regime - who fell less?
                    if len(df_lb_down) >= 10:
                        outperf = df_lb_down[f'Cum_Outperf_{lb}d']
                        avg_outperf = outperf.mean()
                        avg_a = df_lb_down[f'Cum_A_{lb}d'].mean()
                        avg_h = df_lb_down[f'Cum_H_{lb}d'].mean()
                        win_rate = (outperf > 0).mean() * 100  # A fell less = positive outperformance
                        t_stat, p_val = stats.ttest_1samp(outperf.dropna(), 0)
                        
                        multi_horizon_results.append({
                            'Period': f'{lb}D',
                            'Regime': '📉 Both Down',
                            'Observations': len(df_lb_down),
                            'Avg A Return (%)': avg_a,
                            'Avg H Return (%)': avg_h,
                            'A Outperformance (%)': avg_outperf,
                            'A Wins (%)': win_rate,
                            'p-value': p_val,
                            'Sig': '✓' if p_val < 0.05 else ''
                        })
                
                if multi_horizon_results:
                    multi_df = pd.DataFrame(multi_horizon_results)
                    
                    st.markdown("#### Summary: Who Moves More During Co-Movement Periods?")
                    st.dataframe(
                        multi_df.style.format({
                            'Avg A Return (%)': '{:.2f}',
                            'Avg H Return (%)': '{:.2f}',
                            'A Outperformance (%)': '{:.2f}',
                            'A Wins (%)': '{:.1f}',
                            'p-value': '{:.4f}'
                        }).background_gradient(
                            cmap="RdYlGn",
                            subset=['A Outperformance (%)'],
                            vmin=-3, vmax=3
                        ),
                        use_container_width=True, hide_index=True
                    )
                    
                    # Visual comparison
                    col_mh_chart1, col_mh_chart2 = st.columns(2)
                    
                    with col_mh_chart1:
                        # Bar chart: A vs H average returns by regime
                        fig_compare = go.Figure()
                        
                        for regime in ['📈 Both Up', '📉 Both Down']:
                            regime_data = multi_df[multi_df['Regime'] == regime]
                            if len(regime_data) > 0:
                                fig_compare.add_trace(go.Bar(
                                    name=f'{regime} - A',
                                    x=regime_data['Period'],
                                    y=regime_data['Avg A Return (%)'],
                                    marker_color=NIPPON_SUOH if regime == '📈 Both Up' else NIPPON_KOHBAI,
                                    offsetgroup=regime
                                ))
                                fig_compare.add_trace(go.Bar(
                                    name=f'{regime} - H',
                                    x=regime_data['Period'],
                                    y=regime_data['Avg H Return (%)'],
                                    marker_color=NIPPON_RURI if regime == '📈 Both Up' else NIPPON_WASURENAGUSA,
                                    offsetgroup=regime
                                ))
                        
                        fig_compare.update_layout(
                            title="Average Returns: A vs H<br><sub>During Co-Movement Periods</sub>",
                            xaxis_title="Lookback Period",
                            yaxis_title="Average Return (%)",
                            barmode='group',
                            template="seaborn",
                            height=400
                        )
                        st.plotly_chart(fig_compare, use_container_width=True)
                    
                    with col_mh_chart2:
                        # Bar chart: A Outperformance
                        fig_outperf = go.Figure()
                        
                        up_data = multi_df[multi_df['Regime'] == '📈 Both Up']
                        down_data = multi_df[multi_df['Regime'] == '📉 Both Down']
                        
                        if len(up_data) > 0:
                            colors_up = ['#2E7D32' if v > 0 else '#C62828' for v in up_data['A Outperformance (%)']]
                            fig_outperf.add_trace(go.Bar(
                                name='Both Up Periods',
                                x=[f"{p} Up" for p in up_data['Period']],
                                y=up_data['A Outperformance (%)'],
                                marker_color=colors_up,
                                text=[f"{v:.2f}%" for v in up_data['A Outperformance (%)']],
                                textposition='outside'
                            ))
                        
                        if len(down_data) > 0:
                            colors_down = ['#2E7D32' if v > 0 else '#C62828' for v in down_data['A Outperformance (%)']]
                            fig_outperf.add_trace(go.Bar(
                                name='Both Down Periods',
                                x=[f"{p} Down" for p in down_data['Period']],
                                y=down_data['A Outperformance (%)'],
                                marker_color=colors_down,
                                text=[f"{v:.2f}%" for v in down_data['A Outperformance (%)']],
                                textposition='outside'
                            ))
                        
                        fig_outperf.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig_outperf.update_layout(
                            title="A-Share Outperformance (A - H)<br><sub>Green = A wins, Red = H wins</sub>",
                            xaxis_title="Period & Regime",
                            yaxis_title="A Outperformance (%)",
                            showlegend=False,
                            template="seaborn",
                            height=400
                        )
                        st.plotly_chart(fig_outperf, use_container_width=True)
                    
                    # Key insights
                    st.markdown("#### 💡 Key Findings")
                    
                    for lb in selected_lookbacks:
                        lb_data = multi_df[multi_df['Period'] == f'{lb}D']
                        if len(lb_data) == 0:
                            continue
                        
                        st.markdown(f"**{lb}-Day Periods:**")
                        
                        up_row = lb_data[lb_data['Regime'] == '📈 Both Up']
                        down_row = lb_data[lb_data['Regime'] == '📉 Both Down']
                        
                        if len(up_row) > 0:
                            row = up_row.iloc[0]
                            outperf = row['A Outperformance (%)']
                            sig = row['p-value'] < 0.05
                            sig_text = " **(statistically significant)**" if sig else " (not significant)"
                            if outperf > 0:
                                st.success(f"📈 **Bull {lb}D**: When both rally, A gains **{row['Avg A Return (%)']:.2f}%** vs H's **{row['Avg H Return (%)']:.2f}%** → A outperforms by **{outperf:.2f}%**{sig_text}")
                            else:
                                st.error(f"📈 **Bull {lb}D**: When both rally, A gains **{row['Avg A Return (%)']:.2f}%** vs H's **{row['Avg H Return (%)']:.2f}%** → H outperforms by **{-outperf:.2f}%**{sig_text}")
                        
                        if len(down_row) > 0:
                            row = down_row.iloc[0]
                            outperf = row['A Outperformance (%)']
                            sig = row['p-value'] < 0.05
                            sig_text = " **(statistically significant)**" if sig else " (not significant)"
                            if outperf > 0:
                                st.success(f"📉 **Bear {lb}D**: When both fall, A loses **{row['Avg A Return (%)']:.2f}%** vs H's **{row['Avg H Return (%)']:.2f}%** → A falls less by **{outperf:.2f}%**{sig_text}")
                            else:
                                st.error(f"📉 **Bear {lb}D**: When both fall, A loses **{row['Avg A Return (%)']:.2f}%** vs H's **{row['Avg H Return (%)']:.2f}%** → H falls less by **{-outperf:.2f}%**{sig_text}")
                    
                    # Time series visualization
                    st.markdown("---")
                    st.markdown("#### Regime Timeline")
                    
                    viz_lookback = st.selectbox("Visualize regime for lookback:", selected_lookbacks, key="viz_lb")
                    
                    df_viz = df_clean.dropna(subset=[f'Cum_A_{viz_lookback}d', f'Cum_H_{viz_lookback}d'])
                    
                    # Create regime indicator
                    regime_colors = []
                    for idx in df_viz.index:
                        cum_a = df_viz.loc[idx, f'Cum_A_{viz_lookback}d']
                        cum_h = df_viz.loc[idx, f'Cum_H_{viz_lookback}d']
                        if cum_a > 0 and cum_h > 0:
                            regime_colors.append('#2E7D32')
                        elif cum_a < 0 and cum_h < 0:
                            regime_colors.append('#C62828')
                        elif cum_a > 0 and cum_h < 0:
                            regime_colors.append('#1565C0')
                        else:
                            regime_colors.append('#FF8F00')
                    
                    fig_timeline = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                                 row_heights=[0.4, 0.3, 0.3],
                                                 subplot_titles=(f"{viz_lookback}-Day Cumulative Returns", 
                                                                f"{viz_lookback}-Day A Outperformance",
                                                                "AH Spread (%)"))
                    
                    fig_timeline.add_trace(go.Scatter(
                        x=df_viz.index, y=df_viz[f'Cum_A_{viz_lookback}d'],
                        name=f'A {viz_lookback}D Cum Ret', line=dict(color=NIPPON_SUOH, width=1.5)
                    ), row=1, col=1)
                    
                    fig_timeline.add_trace(go.Scatter(
                        x=df_viz.index, y=df_viz[f'Cum_H_{viz_lookback}d'],
                        name=f'H {viz_lookback}D Cum Ret', line=dict(color=NIPPON_RURI, width=1.5)
                    ), row=1, col=1)
                    
                    fig_timeline.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
                    
                    # Outperformance
                    fig_timeline.add_trace(go.Scatter(
                        x=df_viz.index, y=df_viz[f'Cum_Outperf_{viz_lookback}d'],
                        name=f'A Outperformance', 
                        fill='tozeroy',
                        line=dict(color=NIPPON_KOKE, width=1)
                    ), row=2, col=1)
                    fig_timeline.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
                    
                    # Spread with regime coloring
                    fig_timeline.add_trace(go.Scatter(
                        x=df_viz.index, y=df_viz['Spread_Pct'],
                        name='Spread %', 
                        mode='markers+lines',
                        marker=dict(color=regime_colors, size=4),
                        line=dict(color='gray', width=0.5)
                    ), row=3, col=1)
                    
                    fig_timeline.update_layout(height=600, template="seaborn", hovermode="x unified")
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Interpretation (as caption instead of nested expander)
                    st.caption("📖 **Interpretation:** Positive lag = A leads. Green = Both Up, Red = Both Down. A Outperformance = A Return - H Return.")
                else:
                    st.warning("Not enough data for multi-horizon analysis. Try a shorter lookback or more historical data.")
            
            # ===========================================
            # SECTION 5: SCATTER PLOTS & DISTRIBUTION
            # ===========================================
            st.markdown("---")
            st.markdown("### 🔬 Relationship Visualization")
            
            col_sc1, col_sc2 = st.columns(2)
            
            with col_sc1:
                # Scatter: A returns vs Spread change
                fig_scatter1 = go.Figure()
                fig_scatter1.add_trace(go.Scatter(
                    x=df_clean['Ret_A'], y=df_clean['Ret_Spread'],
                    mode='markers', marker=dict(color=NIPPON_SUOH, size=5, opacity=0.5),
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
                    mode='markers', marker=dict(color=NIPPON_RURI, size=5, opacity=0.5),
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
                    
                    fig_regime.add_trace(go.Bar(name='High Vol', x=metrics_names, y=high_vals, marker_color=NIPPON_SUOH))
                    fig_regime.add_trace(go.Bar(name='Low Vol', x=metrics_names, y=low_vals, marker_color=NIPPON_RURI))
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
                                               line=dict(color=NIPPON_SUOH, width=1.5)), row=1, col=1)
            fig_roll_beta.add_trace(go.Scatter(x=roll_dates, y=rolling_beta_h, name='β_H (H-Share)', 
                                               line=dict(color=NIPPON_RURI, width=1.5)), row=1, col=1)
            fig_roll_beta.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
            
            fig_roll_beta.add_trace(go.Scatter(x=roll_dates, y=rolling_r2, name='R²', 
                                               fill='tozeroy', line=dict(color=NIPPON_KOKE, width=1)), row=2, col=1)
            
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

