import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional


def plot_learning_curve(
    rewards: list[float],
    title: str = "Learning Curve",
    window: int = 50,
    save_path: Optional[str] = None,
):
    """에피소드 보상 + 이동평균 학습 곡선."""
    fig, ax = plt.subplots(figsize=(10, 4))
    episodes = np.arange(1, len(rewards) + 1)
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", label="Episode reward")

    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(rewards) + 1), ma, color="steelblue", linewidth=2, label=f"MA-{window}")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_portfolio(
    prices: np.ndarray,
    portfolio_values: np.ndarray,
    trade_log: list[dict],
    ticker: str = "",
    initial_balance: float = 10_000.0,
    save_path: Optional[str] = None,
):
    """주가 + 포트폴리오 가치 + 매수/매도 시점을 2패널로 시각화."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    steps = np.arange(len(prices))

    # 상단: 주가 + 매수/매도 마커
    ax1.plot(steps, prices, color="black", linewidth=1, label="Price")
    buys = [t for t in trade_log if t["action"] == "BUY"]
    sells = [t for t in trade_log if t["action"] == "SELL"]
    if buys:
        ax1.scatter([t["step"] for t in buys], [t["price"] for t in buys],
                    marker="^", color="green", s=60, zorder=5, label="BUY")
    if sells:
        ax1.scatter([t["step"] for t in sells], [t["price"] for t in sells],
                    marker="v", color="red", s=60, zorder=5, label="SELL")
    ax1.set_ylabel("Price")
    ax1.set_title(f"{ticker} — Trading Result" if ticker else "Trading Result")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 하단: 포트폴리오 가치
    ax2.plot(steps[:len(portfolio_values)], portfolio_values, color="royalblue", linewidth=1.5, label="Portfolio")
    ax2.axhline(initial_balance, color="gray", linestyle="--", linewidth=1, label="Initial balance")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_action_distribution(trade_log: list[dict], save_path: Optional[str] = None):
    """매수/매도/홀드 비율 파이차트."""
    total = max(len(trade_log), 1)
    buy_n = sum(1 for t in trade_log if t["action"] == "BUY")
    sell_n = sum(1 for t in trade_log if t["action"] == "SELL")
    hold_n = total - buy_n - sell_n

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [buy_n, sell_n, hold_n],
        labels=["BUY", "SELL", "HOLD"],
        colors=["green", "red", "gray"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Action Distribution")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
