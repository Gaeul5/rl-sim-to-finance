"""
Streamlit 인터랙티브 트레이딩 데모
로컬:  streamlit run app/streamlit_app.py
Colab: !streamlit run app/streamlit_app.py & npx localtunnel --port 8501
"""
import sys, os

# 레포 루트를 sys.path에 추가 (Colab / 로컬 양쪽 지원)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in [ROOT, os.path.join(ROOT, "data"), os.path.join(ROOT, "src", "env"),
            os.path.join(ROOT, "agents"), os.path.join(ROOT, "utils")]:
    if sub not in sys.path:
        sys.path.insert(0, sub)

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")

from fetcher import fetch_stock_data, add_features, train_test_split_df
from trading_env import TradingEnv
from ppo_agent import PPOTradingAgent
from visualizer import plot_portfolio, plot_learning_curve, plot_action_distribution

# ─── 페이지 설정 ────────────────────────────────────────────────────────────
st.set_page_config(page_title="RL Trading Demo", page_icon="📈", layout="wide")
st.title("📈 RL Sim-to-Finance — PPO Trading Demo")

# ─── 사이드바 ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    ticker = st.text_input("종목 티커", value="AAPL", help="예: AAPL, 005930.KS, TSLA")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("학습 시작일", value=__import__("datetime").date(2018, 1, 1))
    with col2:
        end_date = st.date_input("학습 종료일", value=__import__("datetime").date(2024, 1, 1))
    initial_balance = st.number_input("초기 자금 ($)", value=10_000, min_value=1_000, step=1_000)
    total_timesteps = st.select_slider(
        "학습 스텝 수",
        options=[10_000, 50_000, 100_000, 200_000, 500_000],
        value=100_000,
    )
    window_size = st.slider("관측 윈도우 (일)", min_value=5, max_value=60, value=20)
    run_btn = st.button("🚀 학습 & 백테스트 실행", use_container_width=True)

# ─── 메인 ────────────────────────────────────────────────────────────────────
if not run_btn:
    st.info("왼쪽 사이드바에서 설정 후 **학습 & 백테스트 실행** 버튼을 누르세요.")
    st.stop()

# 1. 데이터 수집
with st.spinner(f"{ticker} 데이터 수집 중…"):
    try:
        df_raw = fetch_stock_data(ticker, str(start_date), str(end_date))
        df = add_features(df_raw, window=window_size)
    except ValueError as e:
        st.error(str(e))
        st.stop()

df_train, df_test = train_test_split_df(df, test_ratio=0.2)
st.success(f"데이터 로드 완료 — 학습 {len(df_train)}일 / 테스트 {len(df_test)}일")

col_left, col_right = st.columns([3, 1])
with col_left:
    st.subheader("주가 추이")
    st.line_chart(df["Close"])
with col_right:
    st.metric("학습 기간 수익률",
              f"{(df_train['Close'].iloc[-1] / df_train['Close'].iloc[0] - 1) * 100:.1f}%")
    st.metric("테스트 기간 수익률",
              f"{(df_test['Close'].iloc[-1] / df_test['Close'].iloc[0] - 1) * 100:.1f}%")

# 2. 학습
prices_train = df_train["Close"].values.astype("float32")

with st.spinner(f"PPO 학습 중 ({total_timesteps:,} 스텝)…"):
    agent = PPOTradingAgent(
        env_fn=lambda: TradingEnv(prices_train, window_size=window_size, initial_balance=initial_balance),
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        verbose=0,
        device="auto",
    )
    agent.train(total_timesteps=total_timesteps)

st.success("학습 완료!")

st.subheader("학습 곡선")
if agent.reward_callback.episode_rewards:
    fig_curve = plot_learning_curve(
        agent.reward_callback.episode_rewards,
        title=f"{ticker} PPO Learning Curve",
        window=max(10, len(agent.reward_callback.episode_rewards) // 20),
    )
    st.pyplot(fig_curve)
else:
    st.info("에피소드 보상 데이터가 부족합니다.")

# 3. 백테스트
prices_test = df_test["Close"].values.astype("float32")
test_env = TradingEnv(prices_test, window_size=window_size, initial_balance=initial_balance)

obs, _ = test_env.reset()
done = False
portfolio_values = [float(initial_balance)]

while not done:
    action = agent.predict(obs)
    obs, _, terminated, truncated, info = test_env.step(action)
    portfolio_values.append(info["portfolio_value"])
    done = terminated or truncated

final_value = portfolio_values[-1]
ret_pct = (final_value - initial_balance) / initial_balance * 100

# Buy-and-hold 기준선
bh_return = (float(prices_test[-1]) / float(prices_test[window_size]) - 1) * 100

st.subheader("백테스트 결과")
m1, m2, m3 = st.columns(3)
m1.metric("최종 포트폴리오", f"${final_value:,.0f}", f"{ret_pct:+.2f}%")
m2.metric("Buy & Hold 수익률", f"{bh_return:+.2f}%")
m3.metric("알파 (PPO - B&H)", f"{ret_pct - bh_return:+.2f}%")

fig_port = plot_portfolio(
    prices_test, np.array(portfolio_values), test_env.trade_log,
    ticker=ticker, initial_balance=initial_balance,
)
st.pyplot(fig_port)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("행동 분포")
    fig_dist = plot_action_distribution(test_env.trade_log)
    st.pyplot(fig_dist)

with col_b:
    st.subheader("거래 내역")
    if test_env.trade_log:
        import pandas as pd
        log_df = pd.DataFrame(test_env.trade_log)
        log_df["date"] = df_test.index[log_df["step"].clip(upper=len(df_test) - 1)]
        st.dataframe(log_df[["date", "action", "price"]].reset_index(drop=True), use_container_width=True)
    else:
        st.info("거래 없음")
