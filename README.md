# 📈 Modern Portfolio Visualizer

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://modern-portfolio-visualizer-bpnm3q3dcbxagpqi9twukr.streamlit.app/)

A high-performance quantitative finance tool built with Python and Streamlit. This application automates the process of portfolio optimization using **Modern Portfolio Theory (MPT)** and Monte Carlo simulations to identify the most efficient asset allocation for a given set of stocks.

**🔗 Live Demo:** [View App](https://modern-portfolio-visualizer-bpnm3q3dcbxagpqi9twukr.streamlit.app/)

## ✨ Key Features

- **Real-time Market Data**: Fetches historical adjusted close prices directly from Yahoo Finance API.
- **Monte Carlo Simulation**: Executes up to 10,000 iterations to generate a diverse range of random portfolio weights.
- **Efficient Frontier Mapping**: Visualizes the risk-return trade-off using interactive Plotly scatter plots.
- **Optimization Strategy**: Automatically identifies the "Max Sharpe Ratio" portfolio, highlighting the optimal balance between risk and reward.
- **Risk Analysis**: Generates correlation heatmaps to assess asset diversification.

## 🧠 Mathematical Foundation

This tool is grounded in rigorous mathematical finance principles:

### 1. Risk and Return
The expected portfolio return $E(R_p)$ is calculated as the weighted sum of individual asset returns:
$$E(R_p) = \sum_{i=1}^{n} w_i E(R_i)$$

### 2. Portfolio Volatility
Risk is quantified using the portfolio's standard deviation, accounting for the covariance between assets $\Sigma$:
$$\sigma_p = \sqrt{w^T \Sigma w}$$

### 3. Sharpe Ratio
The optimization engine maximizes the Sharpe Ratio ($S$), assuming a risk-free rate ($R_f$) of 0 for simplicity in this visualizer:
$$S = \frac{E(R_p) - R_f}{\sigma_p}$$

## 🛠️ Tech Stack

- **Frontend/App Framework**: [Streamlit](https://streamlit.io/)
- **Data Analysis**: Pandas, NumPy, SciPy
- **Financial Data**: [yfinance](https://pypi.org/project/yfinance/)
- **Visualization**: Plotly (Interactive Charts)

## 🚀 Local Installation

To run this project locally on your machine:

1. **Clone the repository:**

    ```bash
    git clone [https://github.com/YOUR_USERNAME/modern-portfolio-visualizer.git](https://github.com/YOUR_USERNAME/modern-portfolio-visualizer.git)
    cd modern-portfolio-visualizer
    ```

2. **Set up a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Mac/Linux
    ```


3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```


4. **Launch the app:**

    ```bash
    streamlit run app.py
    ```

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
