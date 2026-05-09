***

# Egypt Multi-Asset AI Portfolio Optimization System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.x-61DAFB.svg)](https://reactjs.org/)
[![Financial-ML](https://img.shields.io/badge/Financial-ML-red.svg)]()

## 📌 Project Overview
This project proposes a computational AI-driven framework for autonomous asset allocation within the Egyptian financial market. By synthesizing **Modern Portfolio Theory (MPT)** with advanced **Machine Learning Pipelines**, the system identifies optimal investment weights across diverse asset classes to maximize returns while mitigating systemic risk.

The system features a quantitative backend for predictive modeling and a **React.js** frontend for interactive portfolio management.

---

## 👥 Team Members
| Name | ID |
| :--- | :--- |
| **Moataz Ashraf** | 23-101290 |
| **Ziad Ahmed Rabie** | 23-101292 |
| **Farah Sultan** | 23-101224 |
| **Mostafa Ibrahim** | 24-101476 |
| **Hassan Ahmed** | 23-101330 |
| **Mohamed Allam** | 23-101291 |

---

## 🏗 System Architecture
The architecture follows a scalable data science pipeline:
1.  **Data Ingestion:** Sourcing high-fidelity data from EGX, CBE, and Yahoo Finance.
2.  **Preprocessing:** Feature engineering, outlier detection, and structural break adjustment.
3.  **AI Modeling:** 
    *   **Regressors:** Forecasting continuous returns (Random Forest, SVR, MLR).
    *   **Classifiers:** Predicting market regimes (XGBoost, SVM, Logistic Regression).
4.  **Portfolio Optimization:** Monte Carlo simulations (10,000 iterations) to plot the **Efficient Frontier** and maximize the **Sharpe Ratio**.
5.  **User Interface:** Interactive React.js dashboard communicating with a Python backend via RESTful APIs.

---

## 📊 Asset Universe
The portfolio is diversified across five primary Egyptian asset classes and instruments:

*   **Equities (Indices):** EGX30, EGX100
*   **Commodities:** Gold (XAU/EGP) 
*   **Fixed Income:** 91-Day Egyptian Treasury Bills (T-Bills)
*   **Real Estate Fund:** Represented by major sector proxies: **SODIC**, **TMG**, and **Palm Hills**.

---

## 🛠 Critical Data Engineering & EDA
A significant portion of this project focuses on **Data Hygiene** in emerging markets:
*   **Structural Break Adjustment:** We identified and programmatically fixed a **-73.34% unadjusted stock split** in the Real Estate proxies (August 14, 2025) to prevent the corruption of rolling technical indicators.
*   **Time-Series Synchronization:** Implemented a **Forward-Fill (limit=3)** mechanism to bridge holiday mismatches between international Gold markets and the local EGX calendar.
*   **Feature Engineering:** Generated quantitative features including Rolling Volatility, MACD Histograms, RSI, and Bollinger Band Width.

---

## 🤖 Machine Learning Pipeline
### Labeling Strategy
To account for transaction costs and market friction, we utilize a strict binary labeling logic:
*   **Class 1 (Up):** Future Return > **0.5% (0.005)**
*   **Class 0 (Down/Flat):** Future Return ≤ 0.5%
*   *This results in a ~65%/35% class imbalance, addressed via cost-sensitive learning (`scale_pos_weight`) in XGBoost.*

### Optimization Theory
The system performs simulations to maximize the **Sharpe Ratio**:
$$\text{Sharpe Ratio} = \frac{E[R_p] - R_f}{\sigma_p}$$
Where $R_f$ is the T-Bill yield and $\sigma_p$ is the portfolio volatility.

---

## 💻 Tech Stack
*   **Analysis:** Python, Pandas, NumPy, Scikit-Learn, XGBoost.
*   **Visualization:** Matplotlib, Seaborn, Plotly.
*   **Backend:** FastAPI / Flask (Python).
*   **Frontend:** React.js, Tailwind CSS, MUI, Recharts.

---

## 🚀 Getting Started

### Installation
1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourusername/egypt-ai-portfolio.git
   ```

2. **Setup Python Environment:**
   ```bash
   pip install -r requirements.txt
   ```

---
