import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime, timedelta

# --- 1. Page Configuration ---
st.set_page_config(page_title="Modern Portfolio Visualizer", layout="wide")

st.title("📈 Modern Portfolio Visualizer")
st.markdown("A portfolio analysis tool based on Modern Portfolio Theory (MPT)")

# --- 2. Sidebar Controls ---
st.sidebar.header("Configuration")

tickers_input = st.sidebar.text_input(
    "Stock Tickers (comma separated)", 
    value="AAPL, MSFT, GOOGL"
)
tickers = [t.strip().upper() for t in tickers_input.split(",")]

col1, col2 = st.sidebar.columns(2)
with col1:
    # Default to past 3 years
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*3))
with col2:
    end_date = st.date_input("End Date", datetime.now())

num_simulations = st.sidebar.slider("Number of Simulations", 1000, 10000, 5000)

# --- 3. Data Processing Functions ---
@st.cache_data
def get_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False)
    
    if df.empty:
        return pd.DataFrame()
    
    # Adj Closeを優先し、なければCloseを使う
    if 'Adj Close' in df.columns:
        data = df['Adj Close']
    elif 'Close' in df.columns:
        data = df['Close']
    else:
        return pd.DataFrame()

    # 銘柄が1つの場合のSeries対策
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    
    # 【ここが重要】
    # 1. まず「列（銘柄）」単位で、全ての値がNaNの銘柄を除外する
    data = data.dropna(axis=1, how='all')
        
    return data.dropna()

# --- 4. Main Application Logic ---
if tickers:
    data = get_data(tickers, start_date, end_date)
    
    if data.empty or len(data.columns) < 1:
        st.error("Failed to fetch data. Please check the ticker symbols and date range.")
    else:
        # Returns Calculation (Log Returns for better statistical properties)
        log_returns = np.log(data / data.shift(1)).dropna()
        
        # --- Visualizations: Price History ---
        st.subheader("Price History (Normalized)")
        cumulative_returns = (1 + data.pct_change()).cumprod()
        fig_price = px.line(cumulative_returns)
        fig_price.update_layout(
            yaxis_title="Cumulative Return (Base 1.0)", 
            hovermode="x unified",
            legend_title="Tickers"
        )
        st.plotly_chart(fig_price, width="stretch")

        # --- Visualizations: Correlation ---
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Correlation Heatmap")
            corr_matrix = log_returns.corr()
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto=True, 
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            st.plotly_chart(fig_corr, width="stretch")

        # --- Monte Carlo Simulation ---
        # Annualization factor (252 trading days)
        avg_returns = log_returns.mean() * 252
        cov_matrix = log_returns.cov() * 252
        
        # Setup results array (3 rows: Return, Volatility, Sharpe Ratio)
        results = np.zeros((3, num_simulations))
        weights_record = []
        
        for i in range(num_simulations):
            # Generate random weights and normalize to 1.0
            weights = np.array(np.random.random(len(tickers)))
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            # Calculate Portfolio Return and Volatility
            p_ret = np.sum(avg_returns * weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            results[0,i] = p_ret
            results[1,i] = p_vol
            results[2,i] = p_ret / p_vol # Sharpe Ratio (Assuming Risk-free rate = 0)

        # Create DataFrame from simulation results
        sim_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio'])
        
        # Identify the Optimal Portfolio (Maximum Sharpe Ratio)
        max_sharpe_idx = sim_df['Sharpe Ratio'].idxmax()
        max_sharpe_port = sim_df.iloc[max_sharpe_idx]
        best_weights = weights_record[max_sharpe_idx]

        with col_right:
            st.subheader("Efficient Frontier")
            fig_ef = px.scatter(
                sim_df, x='Volatility', y='Return', color='Sharpe Ratio',
                labels={'Volatility': 'Annualized Risk', 'Return': 'Annualized Return'},
                template="plotly_dark"
            )
            # Highlight the Max Sharpe Ratio portfolio with a red star
            fig_ef.add_trace(go.Scatter(
                x=[max_sharpe_port[1]], y=[max_sharpe_port[0]],
                mode='markers', marker=dict(color='red', size=15, symbol='star'),
                name='Max Sharpe Ratio'
            ))
            st.plotly_chart(fig_ef, width="stretch")

        # --- Optimal Allocation ---
        st.subheader("Optimal Portfolio Allocation (Max Sharpe Ratio)")
        alloc_df = pd.DataFrame({'Asset': tickers, 'Weight': best_weights})
        fig_alloc = px.bar(
            alloc_df, x='Asset', y='Weight', 
            text_auto='.2%', color='Asset',
            labels={'Weight': 'Portfolio Weight'}
        )
        st.plotly_chart(fig_alloc, width="stretch")

        # --- MPT Explanation ---
        st.divider()
        st.markdown("""
        ### 📖 About Modern Portfolio Theory (MPT)
        Developed by Harry Markowitz, MPT is a mathematical framework for assembling a portfolio of assets such that the expected return is maximized for a given level of risk.
        - **Efficient Frontier**: The set of optimal portfolios that offer the highest expected return for a defined level of risk.
        - **Sharpe Ratio**: A measure that indicates the average return earned in excess of the risk-free rate per unit of volatility. A higher ratio indicates better risk-adjusted performance.
        - **The Power of Diversification**: If assets are not perfectly correlated (correlation < 1), the total risk of the portfolio is lower than the weighted average risk of the individual assets.
        """)

else:
    st.info("Please enter stock tickers (e.g., AAPL, MSFT) in the sidebar to begin analysis.")