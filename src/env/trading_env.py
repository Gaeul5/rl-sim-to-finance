import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    커스텀 주식 트레이딩 환경 (gymnasium 호환)

    상태 (observation):
        - 최근 window_size일 종가 (정규화)
        - 현재 보유 주식 수 (정규화)
        - 현재 현금 잔고 (정규화)

    행동 (action):
        0 = 홀드, 1 = 매수 (전량), 2 = 매도 (전량)

    보상:
        스텝마다 포트폴리오 총가치 변화율
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, prices: np.ndarray, window_size: int = 20, initial_balance: float = 10_000.0):
        super().__init__()
        assert len(prices) > window_size, "prices 길이가 window_size보다 커야 합니다."

        self.prices = prices.astype(np.float32)
        self.window_size = window_size
        self.initial_balance = float(initial_balance)

        obs_dim = window_size + 2  # 종가 window + 보유량 + 잔고
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._max_price = float(self.prices.max())
        self.reset()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        window = self.prices[self.current_step - self.window_size: self.current_step]
        norm_window = window / self._max_price
        norm_shares = np.array([self.shares_held / 100.0], dtype=np.float32)
        norm_cash = np.array([self.cash_balance / self.initial_balance], dtype=np.float32)
        return np.concatenate([norm_window, norm_shares, norm_cash]).astype(np.float32)

    def _portfolio_value(self) -> float:
        return self.cash_balance + self.shares_held * float(self.prices[self.current_step])

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash_balance = self.initial_balance
        self.shares_held = 0
        self.prev_portfolio_value = self.initial_balance
        self.trade_log: list[dict] = []
        return self._get_obs(), {}

    def step(self, action: int):
        price = float(self.prices[self.current_step])

        if action == 1 and self.cash_balance >= price:  # 매수
            shares_to_buy = int(self.cash_balance // price)
            self.shares_held += shares_to_buy
            self.cash_balance -= shares_to_buy * price
            self.trade_log.append({"step": self.current_step, "action": "BUY", "price": price})

        elif action == 2 and self.shares_held > 0:  # 매도
            self.cash_balance += self.shares_held * price
            self.trade_log.append({"step": self.current_step, "action": "SELL", "price": price, "shares": self.shares_held})
            self.shares_held = 0

        portfolio_value = self._portfolio_value()
        reward = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        self.prev_portfolio_value = portfolio_value

        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False

        return self._get_obs(), float(reward), terminated, truncated, {"portfolio_value": portfolio_value}

    def render(self, mode="human"):
        price = float(self.prices[self.current_step])
        pv = self._portfolio_value()
        print(f"Step {self.current_step:4d} | Price: {price:8.2f} | "
              f"Shares: {self.shares_held:4d} | Cash: {self.cash_balance:10.2f} | Portfolio: {pv:10.2f}")
